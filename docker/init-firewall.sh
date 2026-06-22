#!/bin/bash
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    exec sudo "$0" "$@"
fi

iptables -F
iptables -X
iptables -t nat -F
iptables -t nat -X
iptables -t mangle -F
iptables -t mangle -X

ipset destroy allowed-domains 2>/dev/null || true
ipset create allowed-domains hash:net

iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT DROP

iptables -A INPUT -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT

iptables -A OUTPUT -p udp --dport 53 -j ACCEPT
iptables -A INPUT -p udp --sport 53 -j ACCEPT
iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT
iptables -A INPUT -p tcp --sport 53 -j ACCEPT

iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

ALLOWED_DOMAINS_FILE="${ALLOWED_DOMAINS_FILE:-/etc/auto-mode/allowed_domains.txt}"
ALLOWED_DOMAINS=()
if [ -r "$ALLOWED_DOMAINS_FILE" ]; then
    while IFS= read -r line; do
        domain="${line%%#*}"
        domain="$(echo "$domain" | tr -d '[:space:]')"
        [ -n "$domain" ] && ALLOWED_DOMAINS+=("$domain")
    done < "$ALLOWED_DOMAINS_FILE"
fi
if [ "${#ALLOWED_DOMAINS[@]}" -eq 0 ]; then
    echo "WARNING: no domains from $ALLOWED_DOMAINS_FILE; using minimal built-in essentials" >&2
    ALLOWED_DOMAINS=("api.anthropic.com" "claude.ai" "static.crates.io" "github.com")
fi

for domain in "${ALLOWED_DOMAINS[@]}"; do
    echo "Resolving $domain ..."
    ips=$(dig +short A "$domain" 2>/dev/null | grep -E '^[0-9]+\.' || true)
    if [ -z "$ips" ]; then
        echo "  WARNING: no A records for $domain"
        continue
    fi
    while IFS= read -r ip; do
        [ -z "$ip" ] && continue
        ipset add allowed-domains "$ip" 2>/dev/null || true
    done <<< "$ips"
done

GATEWAY=$(ip route show default | awk '/default/ {print $3; exit}')
if [ -n "${GATEWAY:-}" ]; then
    SUBNET=$(ip -4 addr show | awk '/inet /{print $2}' | grep -v '^127' | head -1)
    if [ -n "$SUBNET" ]; then
        iptables -A INPUT -s "$SUBNET" -j ACCEPT
        iptables -A OUTPUT -d "$SUBNET" -j ACCEPT
    fi
fi

iptables -A OUTPUT -m set --match-set allowed-domains dst -p tcp --dport 443 -j ACCEPT
iptables -A OUTPUT -m set --match-set allowed-domains dst -p tcp --dport 80 -j ACCEPT

iptables -A OUTPUT -j LOG --log-prefix "FW-DROP: " --log-level 4 -m limit --limit 5/min
iptables -A OUTPUT -j REJECT --reject-with icmp-port-unreachable

echo "Firewall initialized. Allowed domain count: $(ipset list allowed-domains | grep -c '^[0-9]')"
