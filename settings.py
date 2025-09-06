# Global settings for pipeline behavior.
#
# Enabling PLY validation performs additional integrity checks on point cloud
# files but increases processing time. Disable to speed up pipelines once files
# are trusted.
ENABLE_PLY_VALIDATION = False

# Rounding controls for PLY coordinate fields
# When ``PLY_ROUND_COORDS`` is ``True`` the ``x``, ``y`` and ``z`` values
# in exported PLY files are rounded to ``PLY_ROUND_DECIMALS`` places.
PLY_ROUND_COORDS = False
PLY_ROUND_DECIMALS = 2
