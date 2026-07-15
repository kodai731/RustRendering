# Thyllore Animation -- End User License Agreement (EULA)

> **Status: Draft.** This document is a pre-release draft. The final version
> requires legal review before general availability. Beta participants are
> bound by this draft until it is superseded.

This EULA applies to the **proprietary ONNX model** (the "Component")
bundled with Thyllore Animation, namely:

- The bundled Curve Copilot ONNX model `models/curve_copilot.onnx`.

The Python source files in this distribution (`*.py`) are licensed
**separately** under GPL-3.0-or-later. See `LICENSE.md` for the GPL terms.
The compiled Python extension `thyllore_ml_core` inside
`wheels/thyllore_ml_core-*.whl` is licensed under Apache-2.0, matching its
public source repository. The third-party wheels bundled in `wheels/` carry
their own upstream licenses; see `THIRD_PARTY_LICENSES.md` for attribution.

## 1. Grant of License

Subject to your acceptance of and compliance with this EULA, kodai731
("Licensor") grants you ("Licensee") a non-exclusive, non-transferable,
worldwide, revocable license to:

a. Install and use the Component on machines you personally own or that
   are owned by your employer for the purpose of running the Thyllore
   Animation add-on inside Blender.
b. Use the output produced by the Component (generated FCurves and
   keyframes) in your personal, freelance, or commercial projects,
   including projects you sell or distribute.

You retain full copyright in the output you generate. Licensor claims no
rights over the animations or other artifacts you create.

## 2. Restrictions

You may **not**:

a. Redistribute the Component -- neither the `.onnx` model file itself nor
   the addon ZIP that contains it -- to any third party, whether for free
   or for compensation. Sharing your purchase token or addon ZIP with
   non-purchasers is prohibited.
b. Reverse engineer or otherwise attempt to extract or reconstruct the
   model weights, architecture, or training procedures of the Component,
   except to the extent that applicable law expressly forbids such
   restrictions.
c. Use the output of the Component as training data or evaluation set for
   any competing AI service, animation add-on, or model that targets
   Blender or other 3D content creation tools.
d. Remove, alter, or obscure any copyright notice, license notice, or other
   proprietary marking included with the Component.
e. Sublicense, rent, lease, or lend the Component to a third party.

## 3. Updates

Patch and minor updates (`0.x.y` -> `0.x.(y+1)` or `0.x.y` -> `0.(x+1).0`)
are provided free of charge to existing licensees for at least 12 months
from the date of purchase. Major updates may be distributed as a paid
upgrade; existing licensees retain the right to continue using their
current major version indefinitely.

## 4. Data Transmission

a. **Default (ctx32)**: By default, Curve Copilot runs fully offline with
   the standard prediction context (ctx32). No data is transmitted and no
   telemetry is performed.
b. **High-accuracy prediction (ctx64)**: You may opt in to the
   high-accuracy prediction context (ctx64) in the add-on preferences. In
   exchange, the add-on transmits **anonymized Curve Copilot correction
   data** to Licensor's feedback endpoint: the model input context, the
   model prediction, and the values you subsequently edited, together with
   the add-on version and a random anonymous client id. Records contain no
   object names, bone names, file paths, or personal information; curve
   values are origin-relative, amplitude-normalized, and quantized, and
   timestamps are reduced to day granularity, so your original animation
   cannot be reconstructed.
c. Transmission additionally requires Blender's "Allow Online Access"
   setting. Disabling the opt-in or online access reverts prediction to
   ctx32 immediately and stops all transmission.
d. The optional "Send Feedback" button transmits only the free-form text
   you enter, the add-on version, and the same anonymous client id.
e. Licensor uses the transmitted data solely to improve the Curve Copilot
   model and the add-on, and does not sell, rent, or share it with any
   third party.

## 5. Personal Information

Licensor collects only the email address provided to the payment processor
at the time of purchase. This address is used solely for delivering the
addon ZIP and update notifications. Licensor does not sell, rent, or
otherwise share email addresses with any third party.

## 6. Warranty Disclaimer

THE COMPONENT IS PROVIDED **"AS IS"**, WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT.
LICENSOR DOES NOT WARRANT THAT THE COMPONENT WILL OPERATE UNINTERRUPTED OR
ERROR-FREE, OR THAT THE OUTPUT WILL MEET ANY PARTICULAR QUALITY THRESHOLD.

## 7. Limitation of Liability

TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, IN NO EVENT SHALL
LICENSOR BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR
PUNITIVE DAMAGES, OR FOR LOSS OF PROFITS, REVENUE, DATA, OR BUSINESS
OPPORTUNITY, ARISING OUT OF OR RELATED TO YOUR USE OF OR INABILITY TO USE
THE COMPONENT, WHETHER BASED ON CONTRACT, TORT, NEGLIGENCE, STRICT
LIABILITY, OR OTHERWISE.

LICENSOR'S TOTAL CUMULATIVE LIABILITY ARISING FROM THIS EULA OR YOUR USE OF
THE COMPONENT SHALL NOT EXCEED THE AMOUNT YOU ACTUALLY PAID FOR THE
COMPONENT IN THE 12 MONTHS PRECEDING THE EVENT GIVING RISE TO THE
LIABILITY.

## 8. Governing Law and Forum

This EULA is governed by the laws of Japan, without reference to its
conflict-of-laws principles. Any dispute arising from this EULA shall be
subject to the exclusive jurisdiction of the Tokyo District Court as the
court of first instance.

## 9. Termination

Licensor may terminate this EULA upon written notice if you materially
breach Section 2 (Restrictions). Upon termination, you must cease using
the Component and remove it from all machines under your control. The
restrictions in Section 2, the disclaimers in Sections 6-7, and Section 8
survive termination.

## 10. Refunds

Refund eligibility is determined by the storefront through which you
purchased. Licensor honours the storefront's published refund policy and
does not impose additional restrictions.

## 11. Support

Best-effort support is provided through the Thyllore Animation public
issue tracker. No service-level agreement applies.

## 12. Entire Agreement and Severability

This EULA, together with `LICENSE.md` (covering the GPL portion) and
`THIRD_PARTY_LICENSES.md` (covering bundled third-party packages),
constitutes the entire agreement between you and Licensor concerning the
Component, and supersedes all prior or contemporaneous agreements
regarding the same subject matter.

If any provision of this EULA is found unenforceable, the remainder shall
continue in full force and effect.

## 13. Contact

kodai731 <kodai731@gmail.com>

---

*Last updated: 2026-07-16. This is a pre-release draft. The launch version
will be reviewed by qualified counsel before general availability.*
