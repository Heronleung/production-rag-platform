# CI/CD (Phase 7)

Phase 7 starts with an unprivileged validation workflow. Publishing and deployment remain separate until the registry, trigger, environment, and credentials boundaries are approved.

## Activation note

The repository integration used to prepare Phase 7 can write ordinary repository files but GitHub rejected direct writes under `.github/workflows/` with `403 Resource not accessible by integration`. The reviewed workflow is therefore staged at `ci/github-actions/ci.yml`, and Dependabot configuration at `ci/github-actions/dependabot.yml`.

A repository owner can activate them from a normal authenticated local clone:

```bash
mkdir -p .github/workflows
cp ci/github-actions/ci.yml .github/workflows/ci.yml
cp ci/github-actions/dependabot.yml .github/dependabot.yml
git add .github ci docs/ci-cd.md
git commit -m "ci: activate Phase 7 validation pipeline"
git push origin main
```

Alternatively, grant the repository integration permission to update GitHub Actions workflows and move the files in a later automated commit.

## CI workflow

The workflow runs on pull requests, pushes to `main`, and manual dispatches.

Global permissions are limited to:

```yaml
permissions:
  contents: read
```

It does not receive provider secrets, write packages, or deploy resources.

| Job | Checks |
| --- | --- |
| Python | locked dev sync, Ruff, offline pytest suite |
| Web | `npm ci`, lint, typecheck, tests, production build, production dependency audit |
| Helm | chart lint and immutable-SHA manifest rendering |
| Containers | API and web BuildKit builds with `push: false` |

Container builds only run after the Python, web, and Helm jobs pass.

The offline Python suite retains the SSE request-ID regression that prevents resetting a `ContextVar` token from a different execution context. Integration tests are intentionally separate because they require live Milvus or Ollama services.

## Local parity

```bash
uv sync --frozen --extra dev
uv run ruff check api tests
uv run pytest -m "not integration"

cd web
npm ci
npm run lint
npm run typecheck
npm test
npm run build
npm audit --omit=dev --audit-level=high
cd ..

helm lint deploy/helm/rag-platform \
  -f deploy/helm/rag-platform/values-kind.yaml

TAG="ci-$(git rev-parse HEAD)"
helm template rag deploy/helm/rag-platform \
  --namespace rag \
  -f deploy/helm/rag-platform/values-kind.yaml \
  --set-string "api.image.tag=${TAG}" \
  --set-string "web.image.tag=${TAG}" \
  > /tmp/rag-ci-rendered.yaml

docker build -f Dockerfile.api -t "rag-api:${TAG}" .
docker build -f web/Dockerfile -t "rag-web:${TAG}" web
```

## Publishing boundary

The planned default is GHCR with repository `GITHUB_TOKEN` and only `contents: read` plus `packages: write`. The canonical image identifier will be the full commit SHA or resulting digest. Mutable tags may be added as aliases, but deployment and rollback must use an immutable SHA or digest.

Publishing is not active yet. The following decisions must be approved first:

1. GHCR or another registry.
2. Publish on `main`, version tags, or manual dispatch.
3. BuildKit provenance and SBOM policy.
4. Whether to add keyless Cosign signing.

## Deployment boundary

GitHub-hosted runners cannot safely reach the local kind cluster on Heron's machine. The initial workflow therefore does not deploy.

Options for a later slice are artifact-only publishing, a dedicated self-hosted runner with restricted labels and credentials, or a managed staging cluster protected by a GitHub Environment approval gate.

Do not expose the local Kubernetes API publicly or store kubeconfig files in the repository.

## Supply-chain hardening backlog

- Pin actions to reviewed commit SHAs.
- Add dependency review and repository secret scanning.
- Generate SBOM and provenance attestations during image publication.
- Add a container vulnerability threshold.
- Configure required checks and branch protection after the first green workflow run.
- Record published image digests in the workflow summary and release notes.
