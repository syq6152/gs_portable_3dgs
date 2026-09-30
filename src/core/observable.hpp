/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <functional>
#include <utility>

namespace lfs::core {

    // Observable property with automatic change notification
    template <typename T>
    class Observable {
    public:
        using Callback = std::function<void()>;

        Observable() = default;
        Observable(T initial, Callback cb) : value_(std::move(initial)),
                                             on_change_(std::move(cb)) {}

        Observable& operator=(const T& v) {
            if (!(value_ == v)) {
                value_ = v;
                if (on_change_)
                    on_change_();
            }
            return *this;
        }

        Observable& operator=(T&& v) {
            if (!(value_ == v)) {
                value_ = std::move(v);
                if (on_change_)
                    on_change_();
            }
            return *this;
        }

        operator const T&() const { return value_; }
        [[nodiscard]] const T& get() const { return value_; }
        T& getMutable() { return value_; }

        void notifyChanged() {
            if (on_change_)
                on_change_();
        }
        void setCallback(Callback cb) { on_change_ = std::move(cb); }
        void setQuiet(const T& v) { value_ = v; }
        void setQuiet(T&& v) { value_ = std::move(v); }

    private:
        T value_{};
        Callback on_change_{nullptr};
    };

    // Non-member operators for template argument deduction
    template <typename T>
    bool operator==(const Observable<T>& lhs, const T& rhs) { return lhs.get() == rhs; }

    template <typename T>
    bool operator==(const T& lhs, const Observable<T>& rhs) { return lhs == rhs.get(); }

    template <typename T>
    bool operator!=(const Observable<T>& lhs, const T& rhs) { return lhs.get() != rhs; }

    template <typename T>
    bool operator!=(const T& lhs, const Observable<T>& rhs) { return lhs != rhs.get(); }

    template <typename T>
    auto operator*(const T& lhs, const Observable<T>& rhs) -> decltype(lhs * rhs.get()) {
        return lhs * rhs.get();
    }

    template <typename T>
    auto operator*(const Observable<T>& lhs, const T& rhs) -> decltype(lhs.get() * rhs) {
        return lhs.get() * rhs;
    }

} // namespace lfs::core
