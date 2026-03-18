#!/bin/bash
# ============================================================
# RAG 智能问答 — 远程访问配置脚本
# 让你在外出时通过手机/平板访问运行在电脑上的 RAG 服务
# ============================================================
set -e

PORT="${RAG_SERVER_PORT:-8000}"
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

info()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
error() { echo -e "${RED}[✗]${NC} $1"; }
title() { echo -e "\n${BLUE}=== $1 ===${NC}\n"; }

show_help() {
    cat << 'EOF'
用法: ./remote_access.sh <方案>

可选方案:
  cloudflare    使用 Cloudflare Tunnel（推荐）
                - 免费，无需公网 IP 或端口转发
                - 自动 HTTPS，自定义域名可选
                - 适合长期使用

  tailscale     使用 Tailscale 组网
                - 免费（个人最多 100 台设备）
                - 所有设备处于同一虚拟局域网
                - 无需公网 IP，端到端加密
                - 适合多设备组网

  lan           仅局域网访问（最简单）
                - 同一 WiFi 下的手机直接访问
                - 无需安装任何额外软件
                - 不支持外网访问

  status        查看当前网络状态和访问地址

示例:
  ./remote_access.sh lan           # 查看局域网地址
  ./remote_access.sh cloudflare    # 启动 Cloudflare 隧道
  ./remote_access.sh tailscale     # 配置 Tailscale
  ./remote_access.sh status        # 查看状态
EOF
}

show_status() {
    title "当前网络状态"

    # Local IP
    local_ip=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "未知")
    info "本机 IP: ${local_ip}"
    info "服务端口: ${PORT}"

    # Check if RAG server is running
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/health" 2>/dev/null | grep -q "200"; then
        info "RAG 服务: 运行中"
    else
        error "RAG 服务: 未运行（请先启动: uvicorn api_server:app --host 0.0.0.0 --port ${PORT}）"
    fi

    # Check Cloudflare
    if command -v cloudflared &>/dev/null; then
        if pgrep -x cloudflared &>/dev/null; then
            info "Cloudflare Tunnel: 运行中"
        else
            warn "Cloudflare Tunnel: 已安装但未运行"
        fi
    fi

    # Check Tailscale
    if command -v tailscale &>/dev/null; then
        ts_ip=$(tailscale ip -4 2>/dev/null || echo "")
        if [ -n "$ts_ip" ]; then
            info "Tailscale: 已连接 (${ts_ip})"
            info "Tailscale 访问地址: http://${ts_ip}:${PORT}/chat"
        else
            warn "Tailscale: 已安装但未连接"
        fi
    fi

    echo ""
    info "局域网访问地址: http://${local_ip}:${PORT}/chat"
}

setup_lan() {
    title "局域网访问"

    local_ip=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "未知")

    echo "你的手机和电脑连接同一 WiFi 后，用手机浏览器访问："
    echo ""
    echo -e "  ${GREEN}http://${local_ip}:${PORT}/chat${NC}"
    echo ""
    echo "提示："
    echo "  1. 确保 RAG 服务已启动（host 必须是 0.0.0.0）"
    echo "  2. 如果访问不了，检查电脑防火墙是否放行端口 ${PORT}"
    echo ""
    echo "放行防火墙（Linux）:"
    echo "  sudo ufw allow ${PORT}/tcp    # Ubuntu"
    echo "  sudo firewall-cmd --add-port=${PORT}/tcp --permanent && sudo firewall-cmd --reload  # CentOS"
    echo ""
    echo "放行防火墙（macOS）:"
    echo "  # 系统偏好设置 → 安全性与隐私 → 防火墙 → 防火墙选项 → 允许 Python"
}

setup_cloudflare() {
    title "Cloudflare Tunnel 配置"

    # Check if cloudflared is installed
    if ! command -v cloudflared &>/dev/null; then
        warn "cloudflared 未安装，正在安装..."
        echo ""
        echo "请根据你的系统选择安装方式："
        echo ""
        echo "  macOS:   brew install cloudflared"
        echo "  Ubuntu:  curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o /tmp/cloudflared.deb && sudo dpkg -i /tmp/cloudflared.deb"
        echo "  CentOS:  curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.rpm -o /tmp/cloudflared.rpm && sudo rpm -i /tmp/cloudflared.rpm"
        echo ""
        echo "安装完成后重新运行: ./remote_access.sh cloudflare"
        return 1
    fi

    info "cloudflared 已安装"

    # Quick tunnel (no account needed)
    echo ""
    echo "方案 A: 快速隧道（无需注册，适合临时使用）"
    echo "  运行: cloudflared tunnel --url http://localhost:${PORT}"
    echo "  系统会分配一个临时的 https://xxx.trycloudflare.com 地址"
    echo ""
    echo "方案 B: 命名隧道（需要 Cloudflare 账号，适合长期使用）"
    echo "  1. 登录:        cloudflared tunnel login"
    echo "  2. 创建隧道:    cloudflared tunnel create rag-qa"
    echo "  3. 配置路由:    cloudflared tunnel route dns rag-qa your-domain.com"
    echo "  4. 运行隧道:    cloudflared tunnel run --url http://localhost:${PORT} rag-qa"
    echo ""

    read -p "是否启动快速隧道？[Y/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        info "已跳过。你可以稍后手动运行上述命令。"
    else
        info "启动 Cloudflare 快速隧道..."
        echo "隧道地址将在下方显示，用手机浏览器打开即可访问。"
        echo "按 Ctrl+C 停止隧道。"
        echo ""
        cloudflared tunnel --url "http://localhost:${PORT}"
    fi
}

setup_tailscale() {
    title "Tailscale 组网配置"

    if ! command -v tailscale &>/dev/null; then
        warn "Tailscale 未安装"
        echo ""
        echo "安装方式："
        echo "  Linux:   curl -fsSL https://tailscale.com/install.sh | sh"
        echo "  macOS:   brew install tailscale  或从 App Store 安装"
        echo "  手机端:  App Store / Google Play 搜索 Tailscale"
        echo ""
        echo "安装完成后重新运行: ./remote_access.sh tailscale"
        return 1
    fi

    info "Tailscale 已安装"

    # Check if connected
    ts_ip=$(tailscale ip -4 2>/dev/null || echo "")
    if [ -z "$ts_ip" ]; then
        warn "Tailscale 未连接，正在启动..."
        echo "运行: sudo tailscale up"
        echo "首次使用需要浏览器登录授权。"
        echo ""
        read -p "是否现在启动 Tailscale？[Y/n] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            sudo tailscale up
            ts_ip=$(tailscale ip -4 2>/dev/null || echo "")
        fi
    fi

    if [ -n "$ts_ip" ]; then
        info "Tailscale 已连接"
        echo ""
        echo "手机端操作："
        echo "  1. 安装 Tailscale App（App Store / Google Play）"
        echo "  2. 用同一账号登录"
        echo "  3. 浏览器访问:"
        echo ""
        echo -e "     ${GREEN}http://${ts_ip}:${PORT}/chat${NC}"
        echo ""
        echo "  所有已加入 Tailscale 的设备都可以通过此地址访问。"
    else
        error "Tailscale 连接失败，请检查网络和账号设置。"
    fi
}

# Main
case "${1:-}" in
    cloudflare) setup_cloudflare ;;
    tailscale)  setup_tailscale ;;
    lan)        setup_lan ;;
    status)     show_status ;;
    *)          show_help ;;
esac
