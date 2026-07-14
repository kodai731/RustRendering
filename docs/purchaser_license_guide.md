# Purchaser License Guide

Welcome to the Thyllore add-on for Blender. This guide explains how to obtain, activate, and manage your license key after purchasing on Blender Market.

## 1. How to Obtain Your License Key

After purchasing the Thyllore add-on on [Blender Market](https://blendermarket.com), your license key is issued automatically:

- **Email delivery**: You will receive an email containing your unique license key.
- **Purchases page**: You can also find your license key by logging into your Blender Market account and visiting the **Purchases** section.

Your license key is a unique alphanumeric string that looks like this:

```
XXXX-XXXX-XXXX-XXXX
```

Keep this key safe — you will need it to activate the add-on in Blender.

## 2. How to Enter and Validate Your License Key in Blender

To activate your license in Blender:

1. Open Blender and go to **Edit → Preferences** (or press `Ctrl + ,`).
2. In the Preferences window, select the **Add-ons** tab on the left.
3. Find the **Thyllore** add-on in the list (you can use the search filter).
4. Click the arrow to expand the add-on's settings panel.
5. Locate the **License** section within the panel.
6. Enter your license key in the **License Key** field.
7. Click the **Activate License** button.

### What Happens During Activation

- The add-on sends an HTTPS request to the Thyllore licensing server to verify your key.
- Your device's unique **Device ID** (auto-generated on first install) is bound to this license.
- Upon successful activation, a local token is cached so the add-on does not need to re-authenticate on every launch.
- The license is automatically re-validated once per week while Blender is running.

If activation fails, check your internet connection and ensure the key was entered correctly (no extra spaces or characters).

## 3. Seat Limits

Each Thyllore license has a limited number of **seats** — the maximum number of devices (Blender installations) that can simultaneously use the same license key.

### How Seats Work

- Each unique Device ID consumes one seat when activated.
- The seat count is enforced server-side during activation and weekly re-validation.
- You can see your current Device ID in the License section of the add-on preferences.

### What Happens When the Seat Limit Is Reached

If you try to activate on a new device but all seats are already in use, you will see an error message:

> **All seats for this license are in use.**

To resolve this:

1. On a device you no longer use, open Blender → Preferences → Add-ons → Thyllore.
2. Clear the **License Key** field (delete all characters).
3. This only removes the local cached token — it does not immediately free the seat.
4. Seats are released automatically on the server side after a period of inactivity (no API exists to force release).
5. Once the seat is released, you can activate on your new device.

## 4. Behavior When No License Is Present, Expired, Revoked, or Offline

The Thyllore add-on is designed to **never block your workflow**. If a valid license cannot be verified, the add-on automatically falls back to free-tier functionality (ctx32 context length). The following table summarizes the behavior:

| Situation | Behavior |
|---|---|
| No license key entered | Falls back to free-tier ctx32 functionality |
| License expired | Falls back to free-tier ctx32 functionality |
| License revoked | Falls back to free-tier ctx32 functionality |
| Offline (cannot reach server) | Uses cached token; if cache is missing/expired, falls back to ctx32 |
| Network error during activation | Preserves previous activation state; no change until next retry |

In all cases, the add-on continues to function — you will not lose access to your scenes or be unable to work. The only difference is that advanced features requiring a valid license (such as extended context length beyond ctx32) will be unavailable until the license issue is resolved.

## 5. FAQ

### Can I use my license on multiple computers?

Yes — up to the number of seats included with your license. Each computer has a unique Device ID, and each activated device consumes one seat.

### Do I need an internet connection to use the add-on?

An internet connection is required for initial activation and weekly re-validation. If you are offline, the add-on uses the cached token from the last successful validation. If no cache exists or it has expired, the add-on falls back to free-tier ctx32 functionality.

### What is "ctx32" vs "ctx64"?

The context length (ctx) refers to the amount of context the Curve Copilot model (animation curve prediction) can consider at once. The free tier supports 32 frames of context (ctx32). A valid license unlocks extended context lengths — only 32 and 64 are supported (no "beyond") — enabling more sophisticated animation curve predictions.

### I lost my license key — can I recover it?

Yes. Log into your Blender Market account and visit the **Purchases** page. Your license key is listed there alongside your purchase history. You can also request a resend of the confirmation email from that page.

### Can I reinstall Blender without losing my license?

Yes. Your license is tied to your Device ID, which is stored locally. If you reinstall Blender, the Device ID may change. In that case, simply re-enter your license key in the add-on preferences and click **Activate License**. If all seats are in use, clear the key on an old device first (see Section 3).

### Is my data private?

Yes. The licensing system only transmits your license key and Device ID for verification purposes. No scene data, project files, or personal information is sent to the licensing server.
