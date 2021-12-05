"""Workspace initialization. Consult the WORKSPACE on how to use it."""

# Import external repository rules.
load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")
load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")
load("@rules_cc//cc:repositories.bzl", "rules_cc_dependencies", "rules_cc_toolchains")

def workspace():
    # Initializes Bazel package rules' external dependencies.
    rules_cc_dependencies()
    rules_cc_toolchains()
    rules_pkg_dependencies()
    rules_foreign_cc_dependencies()

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
tf_workspace1 = workspace
