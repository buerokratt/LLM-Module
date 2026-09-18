vault {
  # Local testing: use rag-vault, not bare "vault" — that name collides with the
  # ckb stack on the shared bykstack network and authenticates the wrong Vault.
  address = "http://rag-vault:8200"
  retry {
    num_retries = 5
  }
}

auto_auth {
  method "approle" {
    mount_path = "auth/approle"
    config = {
      role_id_file_path   = "/agent/credentials/llm_role_id"
      secret_id_file_path = "/agent/credentials/llm_secret_id"
      remove_secret_id_file_after_reading = false
    }
  }

  sink "file" {
    config = {
      path = "/agent/llm-token/token"
      mode = 0640
    }
  }
}

cache {
  default_lease_duration = "1h"  # Longer TTL for LLM service
}

listener "tcp" {
  address     = "0.0.0.0:8201"  # Listen on all interfaces
  tls_disable = true
}

api_proxy {
  use_auto_auth_token = true
}
