#!/bin/bash
# Alternative efficient push methods for large repositories

# Method 1: Chunked pushing (for very large repos)
echo "🚀 Method 1: Push in smaller chunks"
git config http.postBuffer 524288000  # 500MB buffer
git config pack.windowMemory 256m     # Reduce memory usage
git config pack.packSizeLimit 2g      # Limit pack size

# Method 2: Push with compression
echo "🚀 Method 2: Push with better compression"
git config core.compression 9         # Maximum compression
git config core.looseCompression 9    # Compress loose objects

# Method 3: Alternative remotes (for backup)
echo "🚀 Method 3: Add multiple remotes"
# git remote add gitlab https://gitlab.com/username/repo.git
# git remote add codeberg https://codeberg.org/username/repo.git

# Method 4: Sparse checkout (for large repos)
echo "🚀 Method 4: Enable sparse checkout for cloning"
# git config core.sparseCheckout true
# echo "app.py" > .git/info/sparse-checkout
# echo "*.py" >> .git/info/sparse-checkout

echo "✅ Git optimization completed!"