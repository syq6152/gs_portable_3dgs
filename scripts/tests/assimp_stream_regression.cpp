// Standalone regression for the Assimp stream reader used by ASCII PLY import.
// Build against the selected Assimp headers and library; no test framework needed.
#include <assimp/IOStreamBuffer.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {
class MemoryStream final : public Assimp::IOStream {
public:
    explicit MemoryStream(std::string bytes) : bytes_(std::move(bytes)) {}

    size_t Read(void* destination, size_t size, size_t count) override {
        if (size == 0) {
            return 0;
        }
        const size_t items = std::min(count, (bytes_.size() - position_) / size);
        std::memcpy(destination, bytes_.data() + position_, items * size);
        position_ += items * size;
        return items;
    }
    size_t Write(const void*, size_t, size_t) override { return 0; }
    aiReturn Seek(size_t offset, aiOrigin origin) override {
        size_t next = 0;
        switch (origin) {
        case aiOrigin_SET:
            next = offset;
            break;
        case aiOrigin_CUR:
            if (offset > bytes_.size() - position_) {
                return aiReturn_FAILURE;
            }
            next = position_ + offset;
            break;
        case aiOrigin_END:
            if (offset > bytes_.size()) {
                return aiReturn_FAILURE;
            }
            next = bytes_.size() - offset;
            break;
        default:
            return aiReturn_FAILURE;
        }
        if (next > bytes_.size()) {
            return aiReturn_FAILURE;
        }
        position_ = next;
        return aiReturn_SUCCESS;
    }
    size_t Tell() const override { return position_; }
    size_t FileSize() const override { return bytes_.size(); }
    void Flush() override {}

private:
    std::string bytes_;
    size_t position_ = 0;
};

struct Fixture {
    const char* name;
    std::string input;
    std::vector<std::string> expected;
};

std::string describe(const std::vector<std::string>& lines) {
    std::string result = "[";
    for (const auto& line : lines) {
        if (result.size() > 1) {
            result += ", ";
        }
        result += '"';
        result += line.size() > 40 ? line.substr(0, 40) + "..." : line;
        result += '"';
    }
    return result + ']';
}

bool check(const Fixture& fixture, size_t blockSize, std::string& detail) {
    MemoryStream stream(fixture.input);
    Assimp::IOStreamBuffer<char> reader(blockSize);
    if (!reader.open(&stream)) {
        detail = "could not open nonempty fixture";
        return false;
    }
    std::vector<char> buffer;
    std::vector<std::string> actual;
    // Bound calls so duplicate records or a stuck EOF become ordinary failures.
    for (size_t calls = 0; calls <= fixture.input.size(); ++calls) {
        if (!reader.getNextLine(buffer)) {
            if (reader.getNextLine(buffer)) {
                detail = "EOF was not stable";
                return false;
            }
            if (actual != fixture.expected) {
                detail = "expected " + describe(fixture.expected) + "; got " + describe(actual);
                return false;
            }
            return true;
        }
        // Assimp consumers read through the appended newline. The vector can
        // contain unused capacity bytes and is not guaranteed to end in NUL.
        const auto end = std::find(buffer.begin(), buffer.end(), '\n');
        if (end == buffer.end()) {
            detail = "successful read omitted its newline sentinel";
            return false;
        }
        actual.emplace_back(buffer.begin(), end);
    }
    detail = "reader did not reach EOF";
    return false;
}

void record(const Fixture& fixture, size_t blockSize, size_t& total, size_t& failures,
        std::vector<size_t>& failedBlocks, std::string& firstFailure) {
    ++total;
    std::string detail;
    if (!check(fixture, blockSize, detail)) {
        ++failures;
        failedBlocks.push_back(blockSize);
        if (firstFailure.empty()) {
            firstFailure = "block=" + std::to_string(blockSize) + ": " + detail;
        }
    }
}

void report(const char* name, const std::vector<size_t>& failedBlocks,
        const std::string& firstFailure) {
    std::cout << name << ": " << (32 - failedBlocks.size()) << "/32 passed";
    if (!failedBlocks.empty()) {
        std::cout << "; failed blocks=";
        for (size_t i = 0; i < failedBlocks.size(); ++i) {
            std::cout << (i == 0 ? "" : ",") << failedBlocks[i];
        }
        std::cout << "\n  " << firstFailure;
    }
    std::cout << '\n';
}
} // namespace

int main() {
    const std::string longLine(257, 'L');
    const std::vector<Fixture> fixtures = {
        {"CRLF records", "a\r\nbb\r\nccc\r\ndddd\r\n", {"a", "bb", "ccc", "dddd"}},
        {"LF records", "a\nbb\nccc\ndddd\n", {"a", "bb", "ccc", "dddd"}},
        {"CR records", "a\rbb\rccc\rdddd\r", {"a", "bb", "ccc", "dddd"}},
        {"mixed line endings", "a\r\nbb\nccc\rdddd\r\n", {"a", "bb", "ccc", "dddd"}},
        {"leading and consecutive blank lines", "\r\n\n\ra\r\n\r\nbb\n\nccc\r\r", {"a", "bb", "ccc"}},
        {"only blank lines", "\r\n\n\r\r\n", {}},
        {"unterminated CRLF last record", "a\r\nbb\r\nccc", {"a", "bb", "ccc"}},
        {"unterminated LF last record", "a\nbb\nccc", {"a", "bb", "ccc"}},
        {"single unterminated record", "payload", {"payload"}},
        {"long terminated record", longLine + "\r\nend\r\n", {longLine, "end"}},
        {"long unterminated record", "start\n" + longLine, {"start", longLine}},
    };
    size_t total = 0;
    size_t failures = 0;
    for (const auto& fixture : fixtures) {
        std::vector<size_t> failedBlocks;
        std::string firstFailure;
        for (size_t block = 1; block <= 32; ++block) {
            record(fixture, block, total, failures, failedBlocks, firstFailure);
        }
        report(fixture.name, failedBlocks, firstFailure);
    }
    // These fixtures put the CR at a block's last byte and LF at the next
    // block's first byte, or put the final data byte exactly at physical EOF.
    for (int kind = 0; kind < 3; ++kind) {
        std::vector<size_t> failedBlocks;
        std::string firstFailure;
        const char* name = kind == 0 ? "CRLF split at every block size"
                                    : kind == 1 ? "unterminated exact-boundary EOF"
                                                : "terminated exact-boundary EOF";
        for (size_t block = 1; block <= 32; ++block) {
            const std::string payload(kind == 0 ? 2 * block - 1 : 2 * block, 'X');
            Fixture fixture{name, payload, {payload}};
            if (kind == 0) {
                fixture.input += "\r\nkeep\r\nlast\r\n";
                fixture.expected.insert(fixture.expected.end(), {"keep", "last"});
            } else if (kind == 2) {
                fixture.input = std::string(2 * block - 1, 'X') + '\n';
                fixture.expected = {std::string(2 * block - 1, 'X')};
            }
            record(fixture, block, total, failures, failedBlocks, firstFailure);
        }
        report(name, failedBlocks, firstFailure);
    }
    std::cout << "TOTAL " << total << "; PASSED " << total - failures
              << "; FAILED " << failures << '\n';
    return failures == 0 ? 0 : 1;
}
