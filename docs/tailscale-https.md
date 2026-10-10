# 部署文档：Tailscale 内网 HTTPS（下游安全接入）

状态：已上线 | 作者：小强 | 日期：2026-10-10 | 适用：网关服务器 + 全部下游客户端

---

## 0. 背景与结果（一句话）

下游客户端在公网，之前走 `http://公网IP:4000` 明文，不安全；且**无域名、客户端多台（5 → ≤100）、都是自有机器**。
选用 **Tailscale**（WireGuard 零信任 mesh VPN）：网关只暴露在 tailnet 内网 `443`，Let's-Encrypt 自动证书，公网零暴露面。

最终形态：

```
客户端 ──HTTPS──▶ https://bedrock-gateway.<tailnet>.ts.net ──▶ tailscaled(443, 仅tailnet) ──proxy──▶ 127.0.0.1:4000 (网关)
```

- 网关入站鉴权不变：仍是 `Authorization: Bearer <BEDROCK_API_KEY>`（或 `x-api-key`）。
- 公网 `4000` 端口在客户端全切完后关闭（见 §4）。

---

## 1. 服务器侧环境：三个必须知道的坑

> 这三条是「环境事实」，不是可选优化。踩过一次后固化在此，从 0 配置时**照 §2 的顺序做即可自动避开**。

### 坑①：阿里云内网 DNS 落在 Tailscale 的 CGNAT 网段里（最致命）

- **事实**：阿里云 ECS 的 DHCP 下发内网 DNS `100.100.2.136 / 100.100.2.138`，这两个 IP 落在 Tailscale 的 CGNAT 网段 `100.64.0.0/10` 内。
- **后果**：`tailscale up` 后，Tailscale 把 `100.64.0.0/10` 整体路由到 `tailscale0` 接口 → 发往阿里内网 DNS 的查询被卷进 tailnet 黑洞 → 网关解析不了上游（AWS/Azure）→ 报 `httpx.ConnectError: [Errno -2] Name or service not known` 或 `Resolving timed out`。
- **关键**：`--accept-dns=false` **拦不住它**。这不是「改 DNS 配置」，而是「DNS 报文被路由劫持」——所以关掉 accept-dns 后症状依旧。
- **解法**：把系统 DNS 换成 CGNAT 之外的公共解析（阿里公共 DNS `223.5.5.5 / 223.6.6.6`），见 §2.2。

### 坑②：网关服务器必须保留自己的公网 DNS + 路由

网关要连 `bedrock-runtime.*.amazonaws.com`、`*.openai.azure.com` 等**公网**上游，所以加入 tailnet 时必须同时关掉两件事：

- `--accept-dns=false`：不让 Tailscale 接管系统 DNS（否则 MagicDNS 抢走解析）。
- `--accept-routes=false`：不接受 tailnet 下发的路由（否则出站可能被改道）。

**副作用**：服务器自己不解析 MagicDNS（`*.ts.net`）。所以**在服务器本机自测 HTTPS** 时，要用 `curl --resolve <域名>:443:<tailscale IP>`，或直接请求 `https://<tailscale IP>`（需忽略证书校验）。

### 坑③：apt 仓库文件名是 `noarmor`，不是 `noarch`

- 正确 URL：`https://pkgs.tailscale.com/stable/ubuntu/jammy.noarmor.gpg` 和 `.../jammy.tailscale-keyring.list`。
- 写错成 `noarch`（或别的 codename）会直接 `404`。codename 对照：Ubuntu 22.04 = `jammy`，24.04 = `noble`。

---

## 2. 服务器侧配置步骤（从 0 到上线）

> 全部以 root 执行。顺序不要乱：**先修 DNS，再 `tailscale up`**。

### 2.1 安装 Tailscale（apt 源法）

```bash
# 1) GPG 公钥（注意 noarmor）
mkdir -p --mode=0755 /usr/share/keyrings
curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/jammy.noarmor.gpg \
  | tee /usr/share/keyrings/tailscale-archive-keyring.gpg >/dev/null

# 2) 核对指纹（应包含 458C A832 957F 5868 = Tailscale Inc. 官方签名密钥）
gpg --show-keys --with-fingerprint /usr/share/keyrings/tailscale-archive-keyring.gpg

# 3) apt 源列表
curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/jammy.tailscale-keyring.list \
  | tee /etc/apt/sources.list.d/tailscale.list

# 4) 安装 + 启动守护进程
apt-get update && apt-get install -y tailscale
systemctl enable --now tailscaled
tailscale version          # 确认可执行（本次 1.104.1）
```

### 2.2 先修 DNS（在 `tailscale up` 之前）

```bash
# 备份原 netplan
cp /etc/netplan/50-cloud-init.yaml /root/netplan-50-cloud-init.yaml.bak

# 固定 DNS 到 CGNAT 之外的公共解析，并关掉 DHCP 的 DNS 下发
# 注意：match.macaddress 保持本机网卡 MAC，不要照抄，只新增 dhcp4-overrides + nameservers 两段
cat > /etc/netplan/50-cloud-init.yaml <<'EOF'
network:
    ethernets:
        eth0:
            dhcp4: true
            dhcp6: false
            dhcp4-overrides:
                use-dns: false
            nameservers:
                addresses:
                    - 223.5.5.5
                    - 223.6.6.6
            match:
                macaddress: 00:16:3e:23:fd:de
            set-name: eth0
    version: 2
EOF

# 防止 cloud-init 在重启后重写回 100.100.2.x（会复发坑①）
mkdir -p /etc/cloud/cloud.cfg.d
printf 'network: {config: disabled}\n' > /etc/cloud/cloud.cfg.d/99-disable-network-config.cfg

# 应用并验证
netplan apply
resolvectl status eth0 | grep -iE 'Current DNS|DNS Servers'   # 应显示 223.5.5.5 / 223.6.6.6
getent ahosts bedrock-runtime.us-east-1.amazonaws.com | head -1
```

### 2.3 登录入网（带两个 accept 关闭旗标）

```bash
tailscale up --accept-dns=false --accept-routes=false
# 打开打印出的 https://login.tailscale.com/a/... 链接 → 浏览器登录 → 授权这台设备
```

> 这两个旗标会作为 pref 持久化（写入 state 文件），重启后自动沿用，无需每次重敲。但**换机从 0 装时仍要记得带上**。

### 2.4 控制台（admin.tailscale.com）两件事

1. **改名**：Machines → 这台机 → Rename → `bedrock-gateway`。
2. **开 HTTPS 证书**：DNS 页 → 确认 **MagicDNS** 与 **HTTPS Certificates** 均开启（不开启则 `tailscale serve --https` 拿不到 Let's-Encrypt 证书会报错）。

### 2.5 挂 HTTPS 反向代理

```bash
tailscale serve --bg --https=443 http://127.0.0.1:4000
tailscale serve status
# 期望输出：
#   https://bedrock-gateway.<tailnet>.ts.net (tailnet only)
#   |-- / proxy http://127.0.0.1:4000
```

`--bg` 后台运行；`serve` 只绑在 `tailscale0` 接口上，不暴露公网。

### 2.6 验证（在服务器本机）

```bash
# 服务器 accept-dns=false 不解析 MagicDNS，用 --resolve 把域名手动指到 tailscale IP
curl -sS --resolve bedrock-gateway.tail9cadc3.ts.net:443:100.89.236.65 \
  https://bedrock-gateway.tail9cadc3.ts.net/health
# 期望 {"status":"ok","version":"...","models":39}

# 验证证书签发方 + SAN
echo | openssl s_client -connect 100.89.236.65:443 \
  -servername bedrock-gateway.tail9cadc3.ts.net 2>/dev/null \
  | openssl x509 -noout -subject -issuer -ext subjectAltName
```

---

## 3. 客户端配置流程

> 客户端要能解析 `bedrock-gateway.<tailnet>.ts.net`，必须用**默认 `tailscale up`**（接受 MagicDNS，不要加 `--accept-dns=false`）。

### 3.1 安装 + 入网（每台一次）

**Linux**（官方一键，覆盖 Debian/Ubuntu/CentOS 等）：

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up    # 打开打印的 URL → 用同一账号登录 → 授权这台设备
```

**macOS**：

- App Store 搜 Tailscale 安装（或 `brew install --cask tailscale`）。
- 打开 App → 菜单栏图标 → Log in → 用同一账号登录授权。

### 3.2 切换 base_url

把客户端现在指向网关的地址改成（**API key 不变**）：

```
https://bedrock-gateway.<tailnet>.ts.net
```

例：原 `http://47.236.104.102:4000` → 改 `https://bedrock-gateway.tail9cadc3.ts.net`。

### 3.3 验证

```bash
curl https://bedrock-gateway.tail9cadc3.ts.net/health
# 期望 {"status":"ok",...}
```

### 3.4 规模化（可选，pre-auth key）

机器较多（≤100）时可生成预授权密钥，免浏览器登录：

1. 控制台 → Settings → Keys → Generate auth key（选 Reusable）。
2. 客户端执行：

```bash
sudo tailscale up --auth-key tskey-xxx
```

预授权密钥下发的设备自动进入同一 tailnet，其余流程不变。

---

## 4. 收尾硬化（Phase 2，客户端全切完后做）

等所有客户端都切到 HTTPS 并验证通过后：

1. **网关绑回回环**：改 `/opt/bedrock-gateway/config.yaml` 的 `server.host: 0.0.0.0` → `127.0.0.1`。
   - ⚠️ 用 `Edit` 工具改会重置文件属主，导致 `bedrock` 用户重启时报 Permission denied。改完后 `chown bedrock:bedrock /opt/bedrock-gateway/config.yaml && chmod 640 ...`，再 `systemctl restart bedrock-gateway`。
2. **关公网端口**：阿里云安全组删除公网 `4000` 入方向规则。
3. 复核：公网 `curl http://47.236.104.102:4000/health` 应不通；tailnet 内 HTTPS 应正常。

---

## 5. 排障 / 回滚速查

| 症状 | 判断 | 处理 |
|---|---|---|
| `tailscale up` 后网关报 `Name or service not known` / `Resolving timed out` | DNS 冲突（坑①复发：netplan 被重写回 `100.100.2.x`） | `resolvectl status eth0` 看 DNS 是否回到 `100.100.2.x`；重跑 §2.2 并 `netplan apply` |
| 回滚 DNS 改动 | 需要还原 | `cp /root/netplan-50-cloud-init.yaml.bak /etc/netplan/50-cloud-init.yaml && netplan apply` |
| 关闭 serve | 停用 HTTPS 反代 | `tailscale serve --https=443 off`（或 `tailscale serve reset` 清空全部） |
| 节点状态排查 | — | `tailscale status`（在线/离线）、`tailscale status --json`（BackendState/Online）、`journalctl -u tailscaled -n 50` |
| 本机自测 MagicDNS 解析失败 | 服务器 accept-dns=false 的正常现象 | 用 `curl --resolve` 或 tailscale IP 直连 |
| `tailscale up` 后 DERP 反复 `no-derp-connection`（Singapore） | 中国大陆到 Tailscale 中继链路抖动 | 通常自愈；若持续，检查出站 443/UDP 41641 是否被拦截 |

---

## 6. 关键值备忘（本次部署）

| 项 | 值 |
|---|---|
| tailnet 后缀 | `tail9cadc3.ts.net`（账号 garming.wu@gmail.com 的自动生成名，换账号会变） |
| 节点名 / MagicDNS | `bedrock-gateway.tail9cadc3.ts.net` |
| tailscale IP | `100.89.236.65` |
| 服务器加入旗标 | `--accept-dns=false --accept-routes=false` |
| DNS | `223.5.5.5` / `223.6.6.6` |
| netplan 备份 | `/root/netplan-50-cloud-init.yaml.bak` |
| 反代命令 | `tailscale serve --bg --https=443 http://127.0.0.1:4000` |
