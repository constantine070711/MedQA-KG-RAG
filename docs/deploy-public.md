# MedQA 公网部署指南

目标：在一台 Linux 云服务器上，用 Docker Compose 运行
Next.js + FastAPI + Neo4j，通过 Caddy 自动申请 HTTPS 证书，只对外开放
`80/443`。

生产入口在 `docker-compose.prod.yml`，反向代理配置在 `deploy/Caddyfile`。

## 1. 先确认部署形态

| 项目 | 建议 |
| --- | --- |
| 服务器 | 4 核 8 GB 内存、60 GB SSD，Ubuntu 22.04/24.04，x86_64 |
| 带宽 | 3-5 Mbps 足够（问答是文本 SSE；图片上传更吃上行） |
| GPU | 不需要。默认 `bge-small-zh-v1.5` 在 CPU 上运行 |
| 最低配置 | 2 核 4 GB 可以启动，但首次提问加载模型时容易 OOM/卡顿，不建议现场演示使用 |
| 网络位置 | 内地服务器 + 域名必须完成 ICP 备案；香港/海外服务器免备案，但内地访问延迟更高 |

不建议把整套后端部署到 Vercel、Railway、Render 这类平台：本项目需要
PyTorch、本地 embedding 模型、Neo4j 常驻进程和长连接 SSE，更适合一台常驻
VPS。

**必须使用 HTTPS**：浏览器的语音识别依赖安全上下文，纯 HTTP 访问时语音输入
不可用（`localhost` 除外）。

## 2. 开放端口

云厂商安全组只放行：

- `22`：SSH，最好限制为自己的出口 IP；
- `80`：Caddy 申请证书并跳转 HTTPS；
- `443`：HTTPS。

不要开放 `3000`、`8000`、`7474`、`7687`。生产覆盖文件已经把 API、Web、
Neo4j 的宿主机端口收回，只保留 Caddy。

## 3. 域名解析

在域名控制台添加 A 记录：

```text
medqa.example.com  ->  服务器公网 IP
```

等解析生效后再启动 Caddy，否则 Let's Encrypt 验证会失败。

如果使用 Cloudflare：

- 首次申请证书时先使用「仅 DNS」（灰云）；
- 证书签发成功后可再开代理（橙云），但服务器防火墙应只允许 Cloudflare
  回源 IP，避免别人绕过 CDN 直连源站。

## 4. 上传代码

仓库地址：

```text
https://github.com/constantine070711/MedQA-KG-RAG.git
```

服务器上：

```bash
sudo mkdir -p /opt/medqa
sudo chown "$USER":"$USER" /opt/medqa
git clone -b v2 https://github.com/constantine070711/MedQA-KG-RAG.git /opt/medqa
cd /opt/medqa
```

仓库里没有模型、索引缓存和 `.env`（被 `.gitignore` 排除），需要从本机补传：

```powershell
# 在本地项目根目录执行
ssh root@SERVER_IP "mkdir -p /opt/medqa/models"
scp -r .\models\bge-small-zh-v1.5 root@SERVER_IP:/opt/medqa/models/
scp -r .\lightrag_cache root@SERVER_IP:/opt/medqa/
scp .\.env.example root@SERVER_IP:/opt/medqa/.env
```

把 `root` 换成你的 SSH 用户；如果服务器禁止 root 登录，使用普通用户并在服务
器命令前加 `sudo`。

说明：

- 只传 `models/bge-small-zh-v1.5`（约 183 MB）即可，默认不使用
  `bge-m3`（约 2.2 GB）和 reranker；
- `lightrag_cache/` 约几 MB，必须传，否则启动后没有可检索的知识索引；
- `lightrag_cache_medkb/` 是实验缓存，生产默认不读取，不必上传；
- 不要上传 `web/node_modules`、`web/.next`、`venv`。

## 5. 安装 Docker

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"
newgrp docker
docker compose version
```

如果服务器拉取 Docker Hub 镜像超时，配置云厂商提供的镜像加速器后再继续。

## 6. 配置 `.env`

```bash
cd /opt/medqa
nano .env
```

至少修改：

```env
DEEPSEEK_API_KEY=<你的 Key>
NEO4J_PASSWORD=<强密码，建议只用字母数字>

MEDQA_DOMAIN=medqa.example.com
PUBLIC_ORIGIN=https://medqa.example.com
```

生成随机密码：

```bash
openssl rand -base64 24
```

注意：

- `.env` 不要提交到 Git，权限设为 `chmod 600 .env`；
- `DEEPSEEK_MODEL` 保持你本地验证可用的模型名，不要为了部署临时更换；
- `MEDQA_DOMAIN` 只写域名，不要带 `https://`；
- `PUBLIC_ORIGIN` 不要带结尾斜杠；
- `NEO4J_URI` 在 `.env` 里即使写 `bolt://localhost:7687` 也没关系，生产
  Compose 会覆盖为 `bolt://neo4j:7687`。

## 7. 初始化 Neo4j

先让后续命令默认同时读取基础配置和生产覆盖：

```bash
echo 'export COMPOSE_FILE=docker-compose.yml:docker-compose.prod.yml' >> ~/.bashrc
source ~/.bashrc
```

构建 API 镜像并启动 Neo4j：

```bash
docker compose build api
docker compose up -d neo4j
```

复用现有 `lightrag_cache/` 导入图谱，**不重新调用 LLM 索引**：

```bash
docker compose --profile init run --rm --entrypoint sh neo4j-init -c "python scripts/neo4j_bootstrap.py && python scripts/import_graphml_to_neo4j.py && python scripts/import_curated_to_neo4j.py && python scripts/rebuild_curated_relations.py"
```

只有在你确实修改了医学文档、需要重建索引时，才运行完整初始化（会消耗
DeepSeek Token，耗时也更长）：

```bash
docker compose --profile init run --rm neo4j-init
```

## 8. 启动全栈

```bash
docker compose up -d --build
docker compose ps
docker compose logs -f caddy
```

首次构建需要下载 PyTorch 等依赖，通常 5-15 分钟。Caddy 日志出现证书签发
成功、各服务 `healthy` 后即可访问：

```text
https://medqa.example.com
```

## 9. 验收清单

```bash
curl -fsS https://medqa.example.com/api/health
curl -fsS -o /dev/null -w '%{http_code}\n' https://medqa.example.com
docker compose logs --tail=100 api web caddy
```

浏览器中逐项确认：

- 首页能打开，证书有效；
- 提问后能看到工作流阶段逐步出现（说明 SSE 没有被缓冲）；
- 回答附带证据/引用与免责声明；
- 图谱证据可用（若不可用会明确显示降级，而不是编造）；
- 图片上传/OCR 可用；
- 语音输入可用（确认是 HTTPS）。

检查图谱数据量：

```bash
docker compose exec neo4j sh -lc 'cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" "MATCH (n) RETURN count(n) AS nodes"'
```

## 10. 日常运维

更新代码：

```bash
cd /opt/medqa
git pull
docker compose up -d --build
```

查看状态与日志：

```bash
docker compose ps
docker stats
docker compose logs -f api web caddy
```

备份（不要加 `-v`，否则会删除 Neo4j 数据卷）：

```bash
docker volume ls | grep neo4j
docker run --rm -v medqa_medqa-neo4j-data:/data:ro -v "$PWD/backups:/backup" alpine tar czf "/backup/neo4j-$(date +%F).tgz" -C /data .
tar czf "backups/lightrag-cache-$(date +%F).tgz" lightrag_cache
```

定期把 `backups/` 拉回本地。Neo4j 数据卷名称以 `docker volume ls` 的实际
输出为准。

## 11. 常见问题

### 证书申请失败

按顺序检查：

1. 域名 A 记录是否已指向服务器；
2. 安全组是否放行 `80/443`；
3. 内地服务器域名是否已完成备案（未备案时运营商可能拦截 80/443）；
4. 是否开了 Cloudflare 橙云但证书模式不匹配（先用灰云验证）；
5. `docker compose logs caddy` 中的 ACME 错误信息。

### 页面能打开，但提问返回 502/失败

```bash
docker compose ps
docker compose logs --tail=200 api
```

确认 API 为 `healthy`，并且 `PUBLIC_ORIGIN` 与浏览器地址完全一致（协议、
域名、端口都要一致）。同域部署下，前端请求的是相对路径 `/api/*`，由 Caddy
转发给 API。

### 返回 429

这是按 IP 的限流，防止公网 API Key 被刷。默认值：

```env
RATE_LIMIT_CHAT_PER_MINUTE=20
RATE_LIMIT_VISION_PER_MINUTE=10
RATE_LIMIT_TOOL_PER_MINUTE=40
RATE_LIMIT_GENERAL_PER_MINUTE=120
```

多人共享同一出口 IP 时，现场演示前可按需调高；修改后重启 API：

```bash
docker compose up -d api
```

### Neo4j 密码改了但登录失败

Neo4j 只在数据卷首次初始化时读取 `NEO4J_PASSWORD`。卷已存在时修改 `.env`
不会改数据库密码。演示环境可删除数据卷后重新初始化（会丢图数据）：

```bash
docker compose down
docker volume rm medqa_medqa-neo4j-data
docker compose up -d neo4j
# 再执行第 7 节的导入命令
```

### 内存不足 / 容器被 OOM Killer 杀掉

```bash
docker stats
free -h
```

可增加 2-4 GB swap，或在 `.env` 中降低 Neo4j 内存：

```env
NEO4J_HEAP=768M
NEO4J_PAGECACHE=256M
```

仍不足时升级到 8 GB 内存。

### 第一次提问特别慢

LightRAG 会在首个请求加载本地 embedding 模型并初始化检索 worker，属于正常
现象；预热一次后后续请求会明显加快。

### 构建慢 / 拉镜像超时

优先配置云厂商的 Docker 镜像加速；PyPI 慢可在构建时使用国内源。不要在
`Dockerfile` 里写死密钥。

### 只有 IP、没有域名，想先跑通

可临时改成 HTTP（仅用于冒烟测试）：

```env
MEDQA_DOMAIN=:80
PUBLIC_ORIGIN=http://SERVER_IP
```

然后 `docker compose up -d`，访问 `http://SERVER_IP`。此模式没有 HTTPS，
语音输入不可用，不建议用于正式演示。

## 12. 安全与成本

- 为线上单独创建 DeepSeek API Key，并在平台设置余额/用量提醒，不要把主 Key
  直接暴露在公网服务中；
- 限流只按单 IP 计数，能挡住脚本刷接口，但不能替代鉴权。若长期公开运行，
  建议增加访问码或登录，并把配额移到 Redis/网关；
- 不要在演示环境上传真实患者隐私数据；
- 不要公开 `7474/7687`，Neo4j Browser 只应在需要时通过 SSH 隧道访问；
- 定期更新基础镜像与依赖，保留 `docker compose logs` 和备份；
- 内地公网服务除 ICP 备案外，可能还需要公安联网备案，按云厂商要求办理。
