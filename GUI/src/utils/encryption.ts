/**
 * RSA Encryption Utility for LLM Connection Secrets
 * 
 * This module provides RSA encryption functionality using the Web Crypto API
 * to encrypt sensitive LLM credentials before storing them in HashiCorp Vault.
 */

const runtime = (window as unknown as { centralVault?: { prefix: string; mount: string } }).centralVault;
const vaultPrefix = runtime?.prefix || import.meta.env.REACT_APP_VAULT_PATH_PREFIX as string;
const vaultMount = (runtime?.mount || import.meta.env.REACT_APP_VAULT_KV_MOUNT || 'secret') as string;
const validSegment = /^[A-Za-z0-9_-]+$/;
if (!vaultPrefix || !vaultPrefix.split('/').every(segment => validSegment.test(segment)) || !validSegment.test(vaultMount)) {
    throw new Error('Invalid Central Vault GUI path configuration');
}

// Cache for the public key
let publicKey: CryptoKey | null = null;

/**
 * Load the RSA public key from Vault via Vite proxy
 */
async function fetchPublicKey() {
    // Return cached key if available
    if (publicKey) {
        return publicKey;
    }

    try {
        // Use Vite proxy to access vault-agent-gui
        // Proxy configured in vite.config.ts: /vault-agent-gui -> http://vault-agent-gui:8202
        const response = await fetch(
            `/vault-agent-gui/v1/${vaultMount}/data/${vaultPrefix}/encryption/public_key`,
            {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            }
        );

        if (!response.ok) {
            throw new Error(`Failed to fetch public key: ${response.status}`);
        }

        const data = await response.json();

        // Extract PEM string from Vault response
        const publicKeyPem = data.data?.data?.key;

        if (!publicKeyPem) {
            throw new Error('Public key not found in response');
        }

        // Convert PEM to CryptoKey object for Web Crypto API
        const cryptoKey = await importPublicKey(publicKeyPem);
        
        // Cache the key
        publicKey = cryptoKey;
        
        return cryptoKey;
        
    } catch (error) {
        console.error('Error fetching public key:', error);
        throw new Error('Failed to load encryption key');
    }
}

// Helper: Convert PEM string to CryptoKey
async function importPublicKey(pemString: string) {
    // Remove PEM headers and whitespace, keep standard base64 encoding
    const pemContents = pemString
        .replace('-----BEGIN PUBLIC KEY-----', '')
        .replace('-----END PUBLIC KEY-----', '')
        .replace(/\s/g, '');  // Remove whitespace and newlines only

    const binaryDer = Uint8Array.from(atob(pemContents), c => c.charCodeAt(0));

    // Import as CryptoKey for encryption
    const cryptoKey = await crypto.subtle.importKey(
        'spki',
        binaryDer.buffer,
        {
            name: 'RSA-OAEP',
            hash: 'SHA-256'
        },
        false,
        ['encrypt']
    );

    return cryptoKey;
}

/**
 * Encrypt a string value using RSA-OAEP
 * @param value - The plaintext value to encrypt
 * @returns Base64-encoded encrypted value
 */
export async function encryptValue(value: string): Promise<string> {
    if (!value) {
        return '';
    }

    try {
        const key = await fetchPublicKey();

        if (!key) {
            throw new Error('Failed to load encryption key');
        }

        // Convert string to ArrayBuffer
        const encoder = new TextEncoder();
        const data = encoder.encode(value);

        // Encrypt the data
        const encryptedData = await window.crypto.subtle.encrypt(
            {
                name: 'RSA-OAEP',
            },
            key,
            data
        );

        // Convert to URL-safe base64 (base64url) for transmission
        // Replace + with -, / with _, and remove trailing =
        const base64 = btoa(String.fromCharCode(...new Uint8Array(encryptedData)))
            .replace(/\+/g, '-')
            .replace(/\//g, '_')
            .replace(/=+$/, '');
        return base64;
    } catch (error) {
        console.error('Error encrypting value:', error);
        throw new Error('Failed to encrypt sensitive data');
    }
}

/**
 * Encrypt an object containing sensitive credentials
 * @param credentials - Object with potentially sensitive string values
 * @returns Object with encrypted values (only encrypts non-empty string values)
 */
export async function encryptCredentials<T extends Record<string, any>>(
    credentials: T
): Promise<T> {
    const encrypted: any = { ...credentials };

    for (const key in credentials) {
        const value = credentials[key];
        // Only encrypt non-empty string values
        if (typeof value === 'string' && value.length > 0) {
            encrypted[key] = await encryptValue(value);
        }
    }

    return encrypted;
}

/**
 * Encrypt LLM credentials before sending to vault
 * @param credentials - The credentials object to encrypt
 */
export async function encryptLLMCredentials(credentials: {
    apiKey?: string;
    secretKey?: string;
    accessKey?: string;
    embeddingAccessKey?: string;
    embeddingSecretKey?: string;
    embeddingAzureApiKey?: string;
}): Promise<typeof credentials> {
    const encrypted: any = { ...credentials };

    // Encrypt AWS credentials
    if (credentials.secretKey) {
        encrypted.secretKey = await encryptValue(credentials.secretKey);
    }
    if (credentials.accessKey) {
        encrypted.accessKey = await encryptValue(credentials.accessKey);
    }

    // Encrypt Azure credentials
    if (credentials.apiKey) {
        encrypted.apiKey = await encryptValue(credentials.apiKey);
    }

    // Encrypt embedding AWS credentials
    if (credentials.embeddingAccessKey) {
        encrypted.embeddingAccessKey = await encryptValue(credentials.embeddingAccessKey);
    }
    if (credentials.embeddingSecretKey) {
        encrypted.embeddingSecretKey = await encryptValue(credentials.embeddingSecretKey);
    }

    // Encrypt embedding Azure credentials
    if (credentials.embeddingAzureApiKey) {
        encrypted.embeddingAzureApiKey = await encryptValue(credentials.embeddingAzureApiKey);
    }

    return encrypted;
}

/**
 * Check if a value appears to be encrypted (base64 encoded)
 * This is a simple heuristic check
 */
export function isEncrypted(value: string): boolean {
    if (!value || value.length === 0) {
        return false;
    }

    // Check if it looks like base64
    const base64Regex = /^[A-Za-z0-9+/]+={0,2}$/;
    return base64Regex.test(value) && value.length > 100; // Encrypted values should be reasonably long
}
