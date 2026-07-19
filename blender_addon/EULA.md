# Thyllore Animation -- End User License Agreement (EULA)

This EULA applies to the **proprietary ONNX model** (the "Component")
bundled with Thyllore Animation, namely:

- The bundled Curve Copilot ONNX model `models/curve_copilot.onnx`.

The Component is proprietary software, Copyright © 2026 kodai731, licensed
(not sold) to you under the terms of this EULA. It was trained solely on
license-clean data: public motion-capture datasets whose licenses permit
commercial use, and the anonymized opt-in feedback data described in
Section 3.

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

## 3. Data Transmission

a. Only if Licensee enables it in the add-on preferences, the add-on
   transmits **anonymized usage data** (the inputs to a feature, the
   feature's output, and Licensee's subsequent corrections, together with
   the add-on version and a random anonymous client id) to Licensor's
   endpoint. Transmitted data contains no object names, file paths, or
   other personal information, and is transformed so that the original
   work cannot be identified.
b. Transmission additionally requires Blender's "Allow Online Access"
   setting. Disabling either setting immediately stops all transmission.
   The add-on's core functionality remains available while transmission
   is disabled.
c. The optional "Send Feedback" button transmits only the free-form text
   Licensee enters, the add-on version, and the same anonymous client id.
d. Licensor uses the transmitted data solely to improve the A.I. model and
   the add-on, and does not sell, rent, or share it with any third party.

## 4. Personal Information

Licensor does not collect any personal information. Personal data provided
at the time of purchase (such as your email address) is collected and
processed by the storefront under its own privacy policy. The private
edition performs no online activation and transmits nothing.

## 5. Warranty Disclaimer

THE COMPONENT IS PROVIDED **"AS IS"**, WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT.
LICENSOR DOES NOT WARRANT THAT THE COMPONENT WILL OPERATE UNINTERRUPTED OR
ERROR-FREE, OR THAT THE OUTPUT WILL MEET ANY PARTICULAR QUALITY THRESHOLD.

## 6. Limitation of Liability

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

## 7. Governing Law and Forum

This EULA is governed by the laws of Japan, without reference to its
conflict-of-laws principles. Any dispute arising from this EULA shall be
subject to the exclusive jurisdiction of the Tokyo District Court as the
court of first instance.

## 8. Termination

This EULA terminates automatically, without notice, if you materially
breach Section 2 (Restrictions). Upon termination, you must cease using
the Component and remove it from all machines under your control. The
restrictions in Section 2, the disclaimers in Sections 5-6, and Section 7
survive termination.

## 9. Refunds

Refund eligibility is determined by the storefront through which you
purchased. Licensor honours the storefront's published refund policy and
does not impose additional restrictions.

## 10. Support

Best-effort support is provided through the Thyllore Animation public
issue tracker. No service-level agreement applies.

## 11. Entire Agreement and Severability

This EULA, together with `LICENSE.md` (covering the GPL portion) and
`THIRD_PARTY_LICENSES.md` (covering bundled third-party packages),
constitutes the entire agreement between you and Licensor concerning the
Component, and supersedes all prior or contemporaneous agreements
regarding the same subject matter.

If any provision of this EULA is found unenforceable, the remainder shall
continue in full force and effect.

## 12. Contact

kodai731 <kodai731@gmail.com>

---

*Last updated: 2026-07-17.*
