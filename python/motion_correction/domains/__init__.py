"""
motion_correction.domains — per-sport correction plugins.

PR-7a scope (spec v3 §0 + PR-7_REVIEW_RESPONSE.md C1): golf only.
NO plugin discovery, NO registry, NO dynamic loading. Callers
import the concrete plugin class directly:

    from motion_correction.domains.golf.plugin import GolfCorrectionPlugin
    plugin = GolfCorrectionPlugin()

Future plugins (tennis, ski, etc.) land in PR-8+. ABC interface
extracted then if real demand materializes.
"""
