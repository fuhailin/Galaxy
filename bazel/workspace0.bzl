"""Third-part workspace initialization. Consult the WORKSPACE on how to use it."""
load("@rules_compressor//tensorflow:workspace2.bzl", rules_compressor_deps = "tf_workspace2")
load("@ps-lite//bazel:workspace2.bzl", pslite_deps = "tf_workspace2")

def workspace():
    rules_compressor_deps()
    pslite_deps()

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
tf_workspace0 = workspace
