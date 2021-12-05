"""Workspace initialization. Consult the WORKSPACE on how to use it."""

# Import third party config rules.
load("//bazel:tensorflow.bzl", "check_bazel_version_at_least")

# Import external repository rules.
load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Import third party repository rules. See go/tfbr-thirdparty.
load("//third_party/eigen3:workspace.bzl", eigen3 = "repo")

def _initialize_third_party():
    """ Load third party repositories.  See above load() statements. """
    eigen3()

# Define all external repositories
def _tf_repositories():
    """All external dependencies for TF builds."""

    # To update any of the dependencies bellow:
    # a) update URL and strip_prefix to the new git commit hash
    # b) get the sha256 hash of the commit by running:
    #    curl -L <url> | sha256sum
    # and update the sha256 with the result.

    tf_http_archive(
        name = "com_github_gflags_gflags",
        sha256 = "34af2f15cf7367513b352bdcd2493ab14ce43692d2dcd9dfc499492966c64dcf",
        strip_prefix = "gflags-2.2.2",
        urls = tf_mirror_urls("https://github.com/gflags/gflags/archive/v2.2.2.tar.gz"),
    )

    tf_http_archive(
        name = "com_github_google_glog",
        sha256 = "62efeb57ff70db9ea2129a16d0f908941e355d09d6d83c9f7b18557c0a7ab59e",
        strip_prefix = "glog-d516278b1cd33cd148e8989aec488b6049a4ca0b",
        urls = tf_mirror_urls("https://github.com/google/glog/archive/d516278b1cd33cd148e8989aec488b6049a4ca0b.zip"),
    )

    http_archive(
        name = "boost",
        sha256 = "7bd7ddceec1a1dfdcbdb3e609b60d01739c38390a5f956385a12f3122049f0ca",
        strip_prefix = "boost_1_76_0",
        build_file = "//third_party:boost.BUILD",
        urls = ["https://boostorg.jfrog.io/artifactory/main/release/1.76.0/source/boost_1_76_0.tar.gz"],
    )

    tf_http_archive(
        name = "nlohmann_json",
        build_file = "//third_party:nlohmann_json.BUILD",
        sha256 = "c377963a95989270c943d522bfefe7b889ef5ed0e1e15d535fd6f6f16ed70732",
        strip_prefix = "json-3.4.0",
        urls = tf_mirror_urls("https://github.com/nlohmann/json/archive/v3.4.0.tar.gz"),
    )

def workspace():
    # Check the bazel version before executing any repository rules, in case
    # those rules rely on the version we require here.
    check_bazel_version_at_least("1.0.0")

    # Import all other repositories. This should happen before initializing
    # any external repositories, because those come with their own
    # dependencies. Those recursive dependencies will only be imported if they
    # don't already exist (at least if the external repository macros were
    # written according to common practice to query native.existing_rule()).
    _tf_repositories()

    # Import third party repositories according to go/tfbr-thirdparty.
    _initialize_third_party()

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
tf_workspace2 = workspace
