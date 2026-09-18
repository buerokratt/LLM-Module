# Vault Agent Configuration for CronManager Service
# This agent provides CronManager with access to encryption keys and write access to secrets

vault {
  # Local testing: use rag-vault, not bare "vault" — that name collides with the
  # ckb stack on the shared bykstack network and authenticates the wrong Vault.
  address = "http://rag-vault:8200"
  retry {
    num_retries = 5
  }
}

# Auto-authentication using AppRole
auto_auth {
  method "approle" {
    mount_path = "auth/approle"
    config = {
      role_id_file_path   = "/agent/credentials/cron_role_id"
      secret_id_file_path = "/agent/credentials/cron_secret_id"
      remove_secret_id_file_after_reading = false
    }
  }

  # Write token to file for CronManager service to use
  sink "file" {
    config = {
      path = "/agent/cron-token/token"
      mode = 0640
    }
  }
}

# Caching configuration
cache {
  default_lease_duration = "30m"  # Medium TTL for CronManager
}

# API proxy listener for CronManager service
listener "tcp" {
  address     = "0.0.0.0:8203"
  tls_disable = true
}

# API proxy configuration
api_proxy {
  use_auto_auth_token = true
}
