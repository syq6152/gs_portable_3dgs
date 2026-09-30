/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "preprocessing/process_runner.hpp"
#include <algorithm>
#include <array>
#include <map>
#include <stdexcept>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace lfs::preprocess {
#ifdef _WIN32
    namespace {
        struct Handle {
            HANDLE value = nullptr;
            ~Handle() { reset(); }
            void reset() {
                if (value && value != INVALID_HANDLE_VALUE)
                    CloseHandle(value);
                value = nullptr;
            }
            Handle() = default;
            Handle(const Handle&) = delete;
            Handle& operator=(const Handle&) = delete;
        };
        std::wstring wide(const std::string& s) {
            if (s.find('\0') != std::string::npos)
                throw std::invalid_argument("NUL in process argument/environment");
            if (s.empty())
                return {};
            int count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), static_cast<int>(s.size()), nullptr, 0);
            if (!count)
                throw std::invalid_argument("Invalid UTF-8 process argument/environment");
            std::wstring out(count, 0);
            MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), static_cast<int>(s.size()), out.data(), count);
            return out;
        }
        std::wstring quote(const std::wstring& arg) {
            std::wstring out = L"\"";
            size_t slashes = 0;
            for (wchar_t c : arg) {
                if (c == L'\\') {
                    ++slashes;
                    continue;
                }
                out.append(c == L'"' ? 2 * slashes + 1 : slashes, L'\\');
                out += c;
                slashes = 0;
            }
            out.append(2 * slashes, L'\\');
            return out + L'"';
        }
        struct CaseInsensitive {
            bool operator()(const std::wstring& a, const std::wstring& b) const { return _wcsicmp(a.c_str(), b.c_str()) < 0; }
        };
        std::vector<wchar_t> environment(const ProcessRequest& request) {
            std::map<std::wstring, std::wstring, CaseInsensitive> values;
            auto* block = GetEnvironmentStringsW();
            if (!block)
                throw std::runtime_error("GetEnvironmentStringsW failed");
            for (auto* p = block; *p; p += wcslen(p) + 1) {
                std::wstring item(p);
                const auto sep = item.find(L'=', 1);
                if (sep != std::wstring::npos)
                    values[item.substr(0, sep)] = item.substr(sep + 1);
            }
            FreeEnvironmentStringsW(block);
            for (const auto& [key, value] : request.environment) {
                if (key.empty() || key.find('=') != std::string::npos)
                    throw std::invalid_argument("Invalid environment key");
                values[wide(key)] = wide(value);
            }
            std::vector<wchar_t> out;
            for (const auto& [key, value] : values) {
                const auto item = key + L"=" + value;
                out.insert(out.end(), item.begin(), item.end());
                out.push_back(0);
            }
            out.push_back(0);
            if (out.size() == 1)
                out.push_back(0);
            return out;
        }
        struct Attributes {
            std::vector<unsigned char> storage;
            LPPROC_THREAD_ATTRIBUTE_LIST list = nullptr;
            ~Attributes() {
                if (list)
                    DeleteProcThreadAttributeList(list);
            }
            bool initialize(HANDLE* handles, size_t bytes) {
                SIZE_T size = 0;
                InitializeProcThreadAttributeList(nullptr, 1, 0, &size);
                storage.resize(size);
                auto* candidate = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(storage.data());
                if (!InitializeProcThreadAttributeList(candidate, 1, 0, &size))
                    return false;
                list = candidate;
                return UpdateProcThreadAttribute(list, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, handles, bytes, nullptr, nullptr);
            }
        };
        void drain(HANDLE pipe, std::string& tail, const ExecutionContext& context, bool callbacks) {
            // Limit each pass so a flooding stream cannot starve stderr, cancellation or timeout.
            std::array<char, 8192> buffer{};
            for (int pass = 0; pass < 32; ++pass) {
                DWORD available = 0, read = 0;
                if (!PeekNamedPipe(pipe, nullptr, 0, nullptr, &available, nullptr)) {
                    if (GetLastError() == ERROR_BROKEN_PIPE)
                        return;
                    throw std::runtime_error("PeekNamedPipe failed");
                }
                if (!available)
                    return;
                if (!ReadFile(pipe, buffer.data(), std::min<DWORD>(available, static_cast<DWORD>(buffer.size())), &read, nullptr))
                    throw std::runtime_error("ReadFile pipe failed");
                if (!read)
                    return;
                tail.append(buffer.data(), read);
                constexpr size_t limit = 256 * 1024;
                if (tail.size() > limit)
                    tail.erase(0, tail.size() - limit);
                if (callbacks && context.on_log)
                    context.on_log(context.stage, std::string(buffer.data(), read));
            }
        }
    } // namespace
#endif

    std::expected<ProcessResult, Error> ProcessRunner::run(const ProcessRequest& request, const ExecutionContext& context) const {
#ifndef _WIN32
        return std::unexpected(Error{.code = ErrorCode::PlatformUnsupported, .message = "Windows runtime required"});
#else
        ProcessResult result;
        auto failure = [&](ErrorCode code, std::string message) -> std::expected<ProcessResult, Error> {
            return std::unexpected(Error{.code = code, .message = std::move(message), .exit_code = result.exit_code, .stdout_tail = result.stdout_tail, .stderr_tail = result.stderr_tail});
        };
        if (context.stop_token.stop_requested())
            return failure(ErrorCode::ProcessCancelled, "Cancelled before launch");
        try {
            if (context.on_progress && !context.on_progress(context.stage, 0.0F, "Starting external process"))
                return failure(ErrorCode::ProcessCancelled, "Progress callback cancelled before launch");
        } catch (...) { return failure(ErrorCode::CallbackFailure, "Progress callback failed before launch"); }
        if (!request.executable.is_absolute() || (!request.working_directory.empty() && !request.working_directory.is_absolute()))
            return failure(ErrorCode::InvalidRequest, "Executable and working directory must be absolute");
        std::error_code ec;
        if (!std::filesystem::is_regular_file(request.executable, ec))
            return failure(ErrorCode::RuntimeMissing, "Runtime executable missing");
        if (context.timeout.count() < 0 || context.stop_grace.count() < 0)
            return failure(ErrorCode::InvalidRequest, "Negative process timeout/grace");

        std::wstring command;
        std::vector<wchar_t> env;
        try {
            command = quote(request.executable.wstring());
            for (const auto& arg : request.arguments)
                command += L" " + quote(wide(arg));
            env = environment(request);
        } catch (const std::exception& e) { return failure(ErrorCode::InvalidRequest, e.what()); }
        if (command.size() >= 32767)
            return failure(ErrorCode::InvalidRequest, "Windows command line too long");

        Handle output_read, output_write, error_read, error_write, input, process, thread, job;
        SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
        auto win_failure = [&](std::string message) { return failure(ErrorCode::ProcessLaunchFailed, message + ": " + std::to_string(GetLastError())); };
        if (!CreatePipe(&output_read.value, &output_write.value, &security, 0) ||
            !CreatePipe(&error_read.value, &error_write.value, &security, 0) ||
            !SetHandleInformation(output_read.value, HANDLE_FLAG_INHERIT, 0) ||
            !SetHandleInformation(error_read.value, HANDLE_FLAG_INHERIT, 0))
            return win_failure("Pipe setup failed");
        input.value = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &security, OPEN_EXISTING, 0, nullptr);
        if (input.value == INVALID_HANDLE_VALUE)
            return win_failure("NUL stdin failed");
        job.value = CreateJobObjectW(nullptr, nullptr);
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if (!job.value || !SetInformationJobObject(job.value, JobObjectExtendedLimitInformation, &limits, sizeof(limits)))
            return win_failure("Job setup failed");

        HANDLE inherited[] = {input.value, output_write.value, error_write.value};
        Attributes attributes;
        if (!attributes.initialize(inherited, sizeof(inherited)))
            return win_failure("Handle inheritance setup failed");
        STARTUPINFOEXW startup{};
        startup.StartupInfo.cb = sizeof(startup);
        startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
        startup.StartupInfo.wShowWindow = SW_HIDE;
        startup.StartupInfo.hStdInput = input.value;
        startup.StartupInfo.hStdOutput = output_write.value;
        startup.StartupInfo.hStdError = error_write.value;
        startup.lpAttributeList = attributes.list;
        PROCESS_INFORMATION info{};
        const auto cwd = request.working_directory.wstring();
        if (!CreateProcessW(request.executable.c_str(), command.data(), nullptr, nullptr, TRUE,
                            CREATE_SUSPENDED | CREATE_NEW_PROCESS_GROUP | CREATE_UNICODE_ENVIRONMENT | EXTENDED_STARTUPINFO_PRESENT,
                            env.data(), cwd.empty() ? nullptr : cwd.c_str(), &startup.StartupInfo, &info))
            return win_failure("CreateProcessW failed");
        process.value = info.hProcess;
        thread.value = info.hThread;
        output_write.reset();
        error_write.reset();
        input.reset();
        if (!AssignProcessToJobObject(job.value, process.value) || ResumeThread(thread.value) == DWORD(-1)) {
            const DWORD error = GetLastError();
            TerminateProcess(process.value, 1);
            WaitForSingleObject(process.value, 5000);
            return failure(ErrorCode::ProcessLaunchFailed, "Cannot attach/resume owned process: " + std::to_string(error));
        }

        const auto start = std::chrono::steady_clock::now();
        std::optional<ErrorCode> stop_reason;
        auto stop_time = start;
        bool killed = false;
        struct Descendants {
            std::vector<HANDLE> handles;
            ~Descendants() {
                for (auto handle : handles)
                    CloseHandle(handle);
            }
            void capture(HANDLE job) {
                std::vector<unsigned char> storage(4096);
                while (true) {
                    auto* ids = reinterpret_cast<JOBOBJECT_BASIC_PROCESS_ID_LIST*>(storage.data());
                    if (QueryInformationJobObject(job, JobObjectBasicProcessIdList, ids, static_cast<DWORD>(storage.size()), nullptr)) {
                        for (DWORD i = 0; i < ids->NumberOfProcessIdsInList; ++i) {
                            auto handle = OpenProcess(SYNCHRONIZE, FALSE, static_cast<DWORD>(ids->ProcessIdList[i]));
                            if (handle)
                                handles.push_back(handle);
                        }
                        return;
                    }
                    if (GetLastError() != ERROR_MORE_DATA || storage.size() >= 16 * 1024 * 1024)
                        return;
                    storage.resize(storage.size() * 2);
                }
            }
        } descendants;
        while (true) {
            try {
                drain(output_read.value, result.stdout_tail, context, !stop_reason);
                drain(error_read.value, result.stderr_tail, context, !stop_reason);
            } catch (...) {
                if (!stop_reason) {
                    stop_reason = ErrorCode::CallbackFailure;
                    stop_time = std::chrono::steady_clock::now();
                }
            }
            const auto now = std::chrono::steady_clock::now();
            if (!stop_reason && context.stop_token.stop_requested()) {
                stop_reason = ErrorCode::ProcessCancelled;
                stop_time = now;
            }
            if (!stop_reason && context.timeout.count() > 0 && now - start >= context.timeout) {
                stop_reason = ErrorCode::ProcessTimeout;
                stop_time = now;
            }
            if (stop_reason) {
                // Best effort cooperative Ctrl+Break; no console or ignoring children fall back to Job termination.
                if (!killed) {
                    descendants.capture(job.value);
                    GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, info.dwProcessId);
                    killed = true;
                }
                if (now - stop_time >= context.stop_grace)
                    TerminateJobObject(job.value, 1);
            }
            const DWORD state = WaitForSingleObject(process.value, 10);
            if (state == WAIT_OBJECT_0)
                break;
            if (state == WAIT_FAILED) {
                TerminateJobObject(job.value, 1);
                WaitForSingleObject(process.value, 5000);
                return win_failure("Process wait failed");
            }
        }
        DWORD exit_code = 1;
        GetExitCodeProcess(process.value, &exit_code);
        result.exit_code = static_cast<int>(exit_code);
        // Also reap descendants left after the direct child exits.
        descendants.capture(job.value);
        TerminateJobObject(job.value, 1);
        const auto reap_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        JOBOBJECT_BASIC_ACCOUNTING_INFORMATION accounting{};
        do {
            if (!QueryInformationJobObject(job.value, JobObjectBasicAccountingInformation, &accounting, sizeof(accounting), nullptr))
                return win_failure("Job accounting failed");
            if (!accounting.ActiveProcesses)
                break;
            if (std::chrono::steady_clock::now() >= reap_deadline)
                return failure(ErrorCode::ProcessFailed, "Process tree did not stop");
            WaitForSingleObject(process.value, 1);
            Sleep(1);
        } while (true);
        for (auto handle : descendants.handles) {
            if (WaitForSingleObject(handle, 5000) != WAIT_OBJECT_0)
                return failure(ErrorCode::ProcessFailed, "Descendant termination did not complete");
        }
        try {
            drain(output_read.value, result.stdout_tail, context, !stop_reason);
            drain(error_read.value, result.stderr_tail, context, !stop_reason);
        } catch (...) { stop_reason = ErrorCode::CallbackFailure; }
        if (stop_reason)
            return failure(*stop_reason, "Process stopped (cancel/timeout/callback); process tree reaped");
        if (result.exit_code != 0)
            return failure(ErrorCode::ProcessFailed, "Process returned nonzero exit code");
        try {
            if (context.on_progress && !context.on_progress(context.stage, 1.0F, "External process complete"))
                return failure(ErrorCode::ProcessCancelled, "Progress callback cancelled completion");
        } catch (...) { return failure(ErrorCode::CallbackFailure, "Progress callback failed at completion"); }
        return result;
#endif
    }
} // namespace lfs::preprocess
