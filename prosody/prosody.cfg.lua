-- Global settings
pidfile = "/var/run/prosody/prosody.pid"

-- Plugin configuration
plugin_paths = { "/usr/lib/prosody/modules-extra" }

-- Load required modules
modules_enabled = {
    -- Core modules (these are built-in and don't need to be explicitly loaded)
    "roster",
    "saslauth",
    "tls",
    "dialback",
    "disco",
    "private",
    "vcard",
    "version",
    "uptime",
    "time",
    "ping",
    "pep",
    "register",
    "admin_adhoc",

    -- Custom monitor module
    "message_monitor",
}

-- Monitor and ban module configuration
message_monitor_endpoint = "http://172.20.0.4:3000/api/messages"
message_monitor_token = "your_secret_token"

-- Global settings
allow_registration = true
c2s_require_encryption = true
s2s_require_encryption = true
allow_unencrypted_plain_auth = false

-- SSL/TLS configuration
ssl = {
    key = "/etc/prosody/certs/prosody.key",
    certificate = "/etc/prosody/certs/prosody.crt",
}

-- Define virtual host
VirtualHost "prosody"
authentication = "internal_plain"
