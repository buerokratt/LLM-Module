# Vault Agent Configuration for GUI Service
# This agent provides GUI with access to public encryption key only

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
      role_id_file_path   = "/agent/credentials/gui_role_id"
      secret_id_file_path = "/agent/credentials/gui_secret_id"
      remove_secret_id_file_after_reading = false
    }
  }

  # Write token to file for GUI service to use
  sink "file" {
    config = {
      path = "/agent/gui-token/token"
      mode = 0640
    }
  }
}

# Caching configuration
cache {
  default_lease_duration = "15m"  # Short-lived tokens for GUI
}

# API proxy listener for GUI service
listener "tcp" {
  address     = "0.0.0.0:8202"
  tls_disable = true
}

# API proxy configuration
api_proxy {
  use_auto_auth_token = true
}
