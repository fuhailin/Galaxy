load("@bazel_skylib//rules:build_test.bzl", "build_test")

build_test(
    name = "build_test",
    targets = [
        "@ps-lite//:ps-lite",
        "@boost",
        "@eigen_archive//:eigen_header_files",
    ],
    visibility = ["//:__pkg__"],
)
