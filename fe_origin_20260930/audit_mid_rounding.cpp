#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "msg/md_msg.h"

// Read-only source diagnostic. Decimal keys follow TWSE's four-decimal BCD
// price encoding. No diagnostic value becomes a model feature or label.
int main(int argc, char **argv)
{
    if (argc != 2 && argc != 3) {
        return 2;
    }
    std::ifstream input(argv[1], std::ios::binary);
    if (!input) {
        return 3;
    }
    constexpr size_t WINDOW = 32;
    std::array<double, WINDOW> starts{}, changes{};
    std::array<int64_t, WINDOW> start_keys{};
    size_t head = 0, count = 0, records = 0, closed_nonzero = 0, quote_mid_noise = 0;
    double last = std::numeric_limits<double>::quiet_NaN();
    double last_ratio = -std::numeric_limits<double>::infinity();
    int64_t last_key = 0;
    const Timestamp cutoff = argc == 3 ? std::stoll(argv[2]) : std::numeric_limits<Timestamp>::max();
    auto reset = [&] {
        last = std::numeric_limits<double>::quiet_NaN();
        last_ratio = -std::numeric_limits<double>::infinity();
        head = count = 0;
    };
    std::cout << "receive_time,seq,type,start_mid,end_mid,start_decimal_key,end_decimal_key,net,path,ratio\n" << std::setprecision(17);
    auto observe = [&](const MsgHeader &header, double mid, int64_t key) {
        if (std::isnan(last)) {
            last = mid;
            last_key = key;
            return;
        }
        if (header.type == MsgType::BOOK && key == last_key && mid != last) {
            ++quote_mid_noise;
        }
        starts[head] = last;
        start_keys[head] = last_key;
        changes[head] = mid - last;
        head = (head + 1) % WINDOW;
        count = std::min(count + 1, WINDOW);
        last = mid;
        last_key = key;
        if (count != WINDOW) {
            return;
        }
        const double net = mid - starts[head];
        double path = 0.0;
        for (const double change : changes) {
            path += std::abs(change);
        }
        last_ratio = net == 0.0 ? std::numeric_limits<double>::quiet_NaN() : path / net;
        if (net == 0.0 || key != start_keys[head]) {
            return;
        }
        ++closed_nonzero;
        if (closed_nonzero <= 20) {
            std::cout << header.time << ',' << header.seq << ',' << static_cast<char>(header.type) << ',' << starts[head] << ',' << mid << ','
                      << start_keys[head] << ',' << key << ',' << net << ',' << path << ',' << path / net << '\n';
        }
    };
    MsgHeader header{};
    while (input.read(reinterpret_cast<char *>(&header), sizeof(header))) {
        if (header.time > cutoff) {
            break;
        }
        ++records;
        if (header.type == MsgType::BOOK) {
            Book5 book{};
            if (!input.read(reinterpret_cast<char *>(&book), sizeof(book))) {
                throw std::runtime_error("Truncated native book");
            }
            const bool valid = book.bid_depth > 0 && book.ask_depth > 0 && book.bid[0] > 0 && book.ask[0] > 0 && book.bid_vol[0] > 0 && book.ask_vol[0] > 0;
            if (is_non_continuous_status(book.status_mask) || !valid) {
                reset();
                continue;
            }
            const int64_t key = std::llround(book.bid[0] * 10000.0) + std::llround(book.ask[0] * 10000.0);
            observe(header, (book.bid[0] + book.ask[0]) / 2.0, key);
        } else if (header.type == MsgType::TRADE) {
            Trade trade{};
            if (!input.read(reinterpret_cast<char *>(&trade), sizeof(trade))) {
                throw std::runtime_error("Truncated native trade");
            }
            if (is_non_continuous_status(trade.status_mask)) {
                reset();
            } else if (!std::isnan(last)) {
                observe(header, last, last_key);
            }
        } else {
            throw std::runtime_error("Unknown native message");
        }
    }
    std::cerr << std::setprecision(17) << "records=" << records << " same_decimal_mid_changes=" << quote_mid_noise
              << " closed_decimal_paths_with_nonzero_float_net=" << closed_nonzero << " sampled_ratio=" << last_ratio << '\n';
}
