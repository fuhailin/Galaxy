load("@bazel_skylib//rules:build_test.bzl", "build_test")

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

build_test(
    name = "build_test",
    targets = [
        "@ps-lite//:ps-lite",
        "@boost",
        "@eigen_archive//:eigen_header_files",
    ],
    visibility = ["//:__pkg__"],
)
