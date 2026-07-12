# Thyllore Animation -- End User License Agreement (EULA)

> **Status: Draft.** This document is a pre-launch draft for the Phase 5.5
> MVP release. The final version requires legal review before General
> Availability. Beta participants are bound by this draft until it is
> superseded.

This EULA applies to the **proprietary binary components** bundled with
Thyllore Animation, namely:

- The compiled Python extension `thyllore_ml_core` (`.pyd` on Windows, `.so`
  on Linux/macOS) inside `wheels/thyllore_ml_core-*.whl`.
- The bundled ONNX model `assets/curve_copilot.onnx`.
- The lazily-downloaded ONNX model `light_t2m.onnx` cached under
  `~/.cache/thyllore/light_t2m/<revision>/` once Text-to-Motion ships.

The Python source files in this distribution (`*.py`) are licensed
**separately** under GPL-2.0-or-later. See `LICENSE.md` for the GPL terms.
The third-party wheels bundled in `wheels/` carry their own upstream
licenses; see `THIRD_PARTY_LICENSES.md` for attribution.

## 1. Grant of License

Subject to your acceptance of and compliance with this EULA, kodai731
("Licensor") grants you ("Licensee") a non-exclusive, non-transferable,
worldwide, revocable license to:

a. Install and use the proprietary binary components on machines you
   personally own or that are owned by your employer for the purpose of
   running the Thyllore Animation add-on inside Blender.
b. Use the output produced by the components (generated FCurves, motion
   tracks, rigs, meshes) in your personal, freelance, or commercial
   projects, including projects you sell or distribute.

You retain full copyright in the output you generate. Licensor claims no
rights over the animations, meshes, or other artifacts you create.

## 2. Restrictions

You may **not**:

a. Redistribute the proprietary binaries -- not the ZIP as a whole, not the
   `.pyd` / `.so` extensions, not the `.onnx` models -- to any third party,
   whether for free or for compensation. Sharing your purchase token or
   addon ZIP with non-purchasers is prohibited.
b. Reverse engineer, decompile, disassemble, or otherwise attempt to derive
   the source code, model weights, or training procedures of the
   proprietary components, except to the extent that applicable law
   expressly forbids such restrictions.
c. Use the output of `light_t2m.onnx` or `curve_copilot.onnx` as training
   data or evaluation set for any competing AI service, animation add-on,
   or model that targets Blender or other 3D content creation tools.
d. Remove, alter, or obscure any copyright notice, license notice, or other
   proprietary marking included with the components.
e. Sublicense, rent, lease, or lend the components to a third party.

## 3. Updates and Versioning

a. **Patch and minor updates** (`0.x.y` -> `0.x.(y+1)` or `0.x.y` ->
   `0.(x+1).0`) are provided free of charge to existing licensees for at
   least 12 months from the date of purchase.
b. **Major updates** (`0.x.y` -> `1.0.0`, `1.x.y` -> `2.0.0`, etc.) may be
   distributed as a paid upgrade. Existing licensees retain the right to
   continue using their current major version indefinitely.
c. Licensor may release security or compliance fixes outside the regular
   cadence; these are always free.

## 4. Phase 6 Migration

When the Thyllore Cloud subscription product launches (Phase 6), MVP
purchasers will receive an automatic invitation to the Cloud Free tier,
based on the email address used at purchase. Acceptance of the Free tier is
optional and does not affect the perpetual license granted by this EULA.

## 5. Warranty Disclaimer

THE COMPONENTS ARE PROVIDED **"AS IS"**, WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT.
LICENSOR DOES NOT WARRANT THAT THE COMPONENTS WILL OPERATE UNINTERRUPTED OR
ERROR-FREE, OR THAT THE OUTPUT WILL MEET ANY PARTICULAR QUALITY THRESHOLD.

## 6. Limitation of Liability

TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, IN NO EVENT SHALL
LICENSOR BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR
PUNITIVE DAMAGES, OR FOR LOSS OF PROFITS, REVENUE, DATA, OR BUSINESS
OPPORTUNITY, ARISING OUT OF OR RELATED TO YOUR USE OF OR INABILITY TO USE
THE COMPONENTS, WHETHER BASED ON CONTRACT, TORT, NEGLIGENCE, STRICT
LIABILITY, OR OTHERWISE.

LICENSOR'S TOTAL CUMULATIVE LIABILITY ARISING FROM THIS EULA OR YOUR USE OF
THE COMPONENTS SHALL NOT EXCEED THE AMOUNT YOU ACTUALLY PAID FOR THE
COMPONENTS IN THE 12 MONTHS PRECEDING THE EVENT GIVING RISE TO THE
LIABILITY.

## 7. Personal Information

Licensor collects only the email address provided to the payment processor
(Blender Market, Gumroad, or Polar.sh) at the time of purchase. This
address is used solely for:

a. Delivering the addon ZIP and update notifications.
b. Inviting Phase 6 SaaS Free tier access (Sec 4).

Licensor does not sell, rent, or otherwise share email addresses with any
third party. Aggregated, non-identifying purchase counts may be published
for marketing or transparency purposes.

The lite Variant performs **no telemetry** and makes no network requests
other than the optional first-time HuggingFace download of
`light_t2m.onnx`, which is initiated only when the user invokes the
Text-to-Motion operator.

## 8. Governing Law and Forum

This EULA is governed by the laws of Japan, without reference to its
conflict-of-laws principles. Any dispute arising from this EULA shall be
subject to the exclusive jurisdiction of the Tokyo District Court as the
court of first instance.

## 9. Termination

Licensor may terminate this EULA upon written notice if you materially
breach Sections 2 (Restrictions). Upon termination, you must cease using
the components and remove them from all machines under your control. The
restrictions in Section 2, the disclaimers in Sections 5-6, and Section 8
survive termination.

## 10. Refunds

Refund eligibility is determined by the storefront through which you
purchased (Blender Market, Gumroad, or Polar.sh). Licensor honours the
storefront's published refund policy and does not impose additional
restrictions.

## 11. Support

Best-effort support is provided through the Thyllore Animation public
issue tracker (URL to be announced before GA). No service-level agreement
applies to MVP purchases. The Phase 6 Cloud subscription includes a
defined SLA.

## 12. Entire Agreement and Severability

This EULA, together with `LICENSE.md` (covering the GPL portion) and
`THIRD_PARTY_LICENSES.md` (covering bundled third-party packages),
constitutes the entire agreement between you and Licensor concerning the
proprietary components, and supersedes all prior or contemporaneous
agreements regarding the same subject matter.

If any provision of this EULA is found unenforceable, the remainder shall
continue in full force and effect.

## 13. Contact

kodai731 <kodai731@gmail.com>

---

*Last updated: 2026-05-01. This is the Phase 5.5 MVP draft. The launch
version will be reviewed by qualified counsel before General Availability.*
