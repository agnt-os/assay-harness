import { createHash } from 'node:crypto';
import { existsSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

type RepoConfig = {
  name: string;
  path: string;
  required: boolean;
  dataArtifacts: string[];
  ignoredDirtyPaths?: string[];
};

type FreshnessConfig = {
  repos: RepoConfig[];
  remote: {
    clustersConfig: string;
    hostsPath: string;
    sshConnectTimeoutSeconds: number;
    remoteRepoPathFallback: string;
  };
};

type DataArtifactReport = {
  path: string;
  exists: boolean;
  sizeBytes?: number;
  modifiedAt?: string;
  sha256?: string;
};

type RepoReport = {
  name: string;
  path: string;
  ok: boolean;
  errors: string[];
  branch?: string;
  upstream?: string;
  head?: string;
  upstreamHead?: string;
  ahead?: number;
  behind?: number;
  dirty?: boolean;
  dataArtifacts: DataArtifactReport[];
};

type NodeReport = {
  node: string;
  ssh: string;
  online: boolean;
  ok: boolean;
  errors: string[];
  repos: RepoReport[];
};

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const configPath = path.join(repoRoot, 'config/fleet-freshness.json');

function parseArgs(argv: string[]) {
  const args = new Set(argv);
  const valueAfter = (flag: string) => {
    const index = argv.indexOf(flag);
    return index >= 0 ? argv[index + 1] : undefined;
  };

  return {
    json: args.has('--json'),
    remote: args.has('--remote'),
    noFetch: args.has('--no-fetch'),
    repoList: valueAfter('--repos')?.split(',').map((name) => name.trim()).filter(Boolean),
    hostList: valueAfter('--hosts')?.split(',').map((name) => name.trim()).filter(Boolean)
  };
}

function run(command: string, args: string[], cwd: string) {
  const result = spawnSync(command, args, {
    cwd,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe']
  });

  return {
    status: result.status ?? 1,
    stdout: result.stdout.trim(),
    stderr: result.stderr.trim()
  };
}

function git(cwd: string, args: string[]) {
  return run('git', args, cwd);
}

function loadJson<T>(filePath: string): T {
  return JSON.parse(readFileSync(filePath, 'utf8')) as T;
}

function fingerprintArtifact(repoPath: string, relativeArtifactPath: string): DataArtifactReport {
  const artifactPath = path.resolve(repoPath, relativeArtifactPath);
  if (!existsSync(artifactPath)) {
    return { path: relativeArtifactPath, exists: false };
  }

  const stats = statSync(artifactPath);
  const hash = createHash('sha256').update(readFileSync(artifactPath)).digest('hex');

  return {
    path: relativeArtifactPath,
    exists: true,
    sizeBytes: stats.size,
    modifiedAt: stats.mtime.toISOString(),
    sha256: hash
  };
}

function dirtyPathFromStatusLine(line: string) {
  const rawPath = line.slice(3);
  const renameSeparator = ' -> ';
  const normalizedPath = rawPath.includes(renameSeparator)
    ? rawPath.slice(rawPath.indexOf(renameSeparator) + renameSeparator.length)
    : rawPath;
  return normalizedPath.replace(/^"|"$/g, '');
}

function checkRepo(repo: RepoConfig, options: { noFetch: boolean }): RepoReport {
  const absoluteRepoPath = path.resolve(repoRoot, repo.path);
  const report: RepoReport = {
    name: repo.name,
    path: absoluteRepoPath,
    ok: true,
    errors: [],
    dataArtifacts: []
  };

  if (!existsSync(absoluteRepoPath)) {
    report.ok = !repo.required;
    report.errors.push(`repo path does not exist: ${absoluteRepoPath}`);
    return report;
  }

  const isGitRepo = git(absoluteRepoPath, ['rev-parse', '--is-inside-work-tree']);
  if (isGitRepo.status !== 0 || isGitRepo.stdout !== 'true') {
    report.ok = !repo.required;
    report.errors.push('path is not a git worktree');
    return report;
  }

  if (!options.noFetch) {
    const fetch = git(absoluteRepoPath, ['fetch', '--prune']);
    if (fetch.status !== 0) {
      report.errors.push(`git fetch --prune failed: ${fetch.stderr || fetch.stdout}`);
    }
  }

  const branch = git(absoluteRepoPath, ['branch', '--show-current']);
  const head = git(absoluteRepoPath, ['rev-parse', 'HEAD']);
  const upstream = git(absoluteRepoPath, ['rev-parse', '--abbrev-ref', '--symbolic-full-name', '@{upstream}']);
  const dirty = git(absoluteRepoPath, ['status', '--porcelain']);

  report.branch = branch.stdout || 'DETACHED';
  report.head = head.stdout || undefined;
  const ignoredDirtyPaths = new Set(repo.ignoredDirtyPaths ?? []);
  const dirtyPaths = dirty.stdout
    .split('\n')
    .map((line) => line.trimEnd())
    .filter(Boolean)
    .map(dirtyPathFromStatusLine)
    .filter((dirtyPath) => !ignoredDirtyPaths.has(dirtyPath));
  report.dirty = dirtyPaths.length > 0;

  if (upstream.status !== 0) {
    report.errors.push('branch has no upstream tracking branch');
  } else {
    report.upstream = upstream.stdout;
    const upstreamHead = git(absoluteRepoPath, ['rev-parse', '@{upstream}']);
    const counts = git(absoluteRepoPath, ['rev-list', '--left-right', '--count', 'HEAD...@{upstream}']);

    report.upstreamHead = upstreamHead.stdout || undefined;

    if (counts.status === 0) {
      const [aheadRaw, behindRaw] = counts.stdout.split(/\s+/);
      report.ahead = Number(aheadRaw);
      report.behind = Number(behindRaw);
      if (report.behind > 0) {
        report.errors.push(`branch is behind ${report.upstream} by ${report.behind} commit(s)`);
      }
    } else {
      report.errors.push(`could not compare with upstream: ${counts.stderr || counts.stdout}`);
    }
  }

  if (report.dirty) {
    report.errors.push(`worktree has local modifications: ${dirtyPaths.join(', ')}`);
  }

  report.dataArtifacts = repo.dataArtifacts.map((artifact) => fingerprintArtifact(absoluteRepoPath, artifact));
  for (const artifact of report.dataArtifacts) {
    if (!artifact.exists) {
      report.errors.push(`required data artifact is missing: ${artifact.path}`);
    }
  }
  report.ok = report.errors.length === 0;
  return report;
}

function localReport(config: FreshnessConfig, options: { noFetch: boolean; repoList?: string[] }) {
  const wanted = new Set(options.repoList ?? config.repos.map((repo) => repo.name));
  const repos = config.repos
    .filter((repo) => wanted.has(repo.name))
    .map((repo) => checkRepo(repo, { noFetch: options.noFetch }));

  return {
    node: 'local',
    ok: repos.every((repo) => repo.ok),
    repos
  };
}

function getByPath(value: unknown, dottedPath: string): unknown {
  return dottedPath.split('.').reduce<unknown>((current, part) => {
    if (current && typeof current === 'object' && part in current) {
      return (current as Record<string, unknown>)[part];
    }
    return undefined;
  }, value);
}

function loadRemoteHosts(config: FreshnessConfig, hostList?: string[]) {
  const clustersPath = path.resolve(repoRoot, config.remote.clustersConfig);
  const clusters = loadJson<Record<string, unknown>>(clustersPath);
  const rawHosts = getByPath(clusters, config.remote.hostsPath);
  const wanted = hostList ? new Set(hostList) : undefined;

  if (!rawHosts || typeof rawHosts !== 'object') {
    throw new Error(`could not load hosts from ${config.remote.clustersConfig}:${config.remote.hostsPath}`);
  }

  return Object.entries(rawHosts as Record<string, Record<string, unknown>>)
    .filter(([name]) => !wanted || wanted.has(name))
    .map(([name, hostConfig]) => ({
      name,
      ssh: String(hostConfig.ssh ?? name),
      repoPath: String(hostConfig.training_remote_dir ?? config.remote.remoteRepoPathFallback)
    }));
}

function checkRemoteNode(config: FreshnessConfig, host: { name: string; ssh: string; repoPath: string }): NodeReport {
  const remoteCommand = [
    `cd ${JSON.stringify(host.repoPath)}`,
    'npm run -s fleet:freshness -- --json'
  ].join(' && ');

  const result = spawnSync('ssh', [
    '-o',
    'BatchMode=yes',
    '-o',
    `ConnectTimeout=${config.remote.sshConnectTimeoutSeconds}`,
    host.ssh,
    remoteCommand
  ], {
    cwd: repoRoot,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe']
  });

  if (result.status !== 0) {
    return {
      node: host.name,
      ssh: host.ssh,
      online: false,
      ok: true,
      errors: [`offline or unavailable: ${result.stderr.trim() || result.stdout.trim()}`],
      repos: []
    };
  }

  try {
    const parsed = JSON.parse(result.stdout.trim()) as { ok: boolean; repos: RepoReport[] };
    return {
      node: host.name,
      ssh: host.ssh,
      online: true,
      ok: parsed.ok,
      errors: [],
      repos: parsed.repos
    };
  } catch (error) {
    return {
      node: host.name,
      ssh: host.ssh,
      online: true,
      ok: false,
      errors: [`could not parse remote freshness report: ${(error as Error).message}`],
      repos: []
    };
  }
}

function printLocal(report: ReturnType<typeof localReport>) {
  console.log(`Fleet freshness: ${report.ok ? 'PASS' : 'FAIL'}`);
  for (const repo of report.repos) {
    const state = repo.ok ? 'PASS' : 'FAIL';
    const upstream = repo.upstream ? ` -> ${repo.upstream}` : '';
    const delta = `ahead=${repo.ahead ?? 'n/a'} behind=${repo.behind ?? 'n/a'}`;
    console.log(`${state} ${repo.name}: ${repo.branch ?? 'unknown'}${upstream} ${delta}`);

    for (const artifact of repo.dataArtifacts) {
      if (artifact.exists) {
        console.log(`  data ${artifact.path}: sha256=${artifact.sha256} size=${artifact.sizeBytes} modified=${artifact.modifiedAt}`);
      } else {
        console.log(`  data ${artifact.path}: missing`);
      }
    }

    for (const error of repo.errors) {
      console.log(`  error: ${error}`);
    }
  }
}

function printRemote(nodes: NodeReport[]) {
  console.log(`Online fleet freshness: ${nodes.every((node) => node.ok) ? 'PASS' : 'FAIL'}`);
  for (const node of nodes) {
    if (!node.online) {
      console.log(`SKIP ${node.node} (${node.ssh}): offline`);
      continue;
    }

    console.log(`${node.ok ? 'PASS' : 'FAIL'} ${node.node} (${node.ssh})`);
    for (const repo of node.repos) {
      console.log(`  ${repo.ok ? 'PASS' : 'FAIL'} ${repo.name}: ${repo.branch ?? 'unknown'} -> ${repo.upstream ?? 'no-upstream'} behind=${repo.behind ?? 'n/a'} dirty=${repo.dirty ?? 'n/a'}`);
      for (const error of repo.errors) {
        console.log(`    error: ${error}`);
      }
    }
  }
}

const args = parseArgs(process.argv.slice(2));
const config = loadJson<FreshnessConfig>(configPath);

if (args.remote) {
  const hosts = loadRemoteHosts(config, args.hostList);
  const nodes = hosts.map((host) => checkRemoteNode(config, host));
  if (args.json) {
    console.log(JSON.stringify({ ok: nodes.every((node) => node.ok), nodes }, null, 2));
  } else {
    printRemote(nodes);
  }
  process.exit(nodes.every((node) => node.ok) ? 0 : 1);
}

const report = localReport(config, { noFetch: args.noFetch, repoList: args.repoList });
if (args.json) {
  console.log(JSON.stringify(report, null, 2));
} else {
  printLocal(report);
}
process.exit(report.ok ? 0 : 1);
