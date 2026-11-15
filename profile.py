# Tiny timing helper.
#
#   from profile import perf_start, perf_end, perf_report
#
#   perf_start("step")
#   perf_start("forward")
#   ...
#   perf_end("forward")      # prints "  forward: 3.214 ms"  (indented: it's inside "step")
#   perf_end("step")         # prints "step: 5.002 ms"
#
#   perf_report()            # optional: total / count / average per id, slowest first
#
# Pass quiet=True to perf_end to skip the per-call print (handy inside loops) and
# only collect totals for perf_report().

import time

_open = []      # stack of (id, start_time) for markers that are currently running
_totals = {}    # id -> [total_seconds, count]


def perf_start(id):
    _open.append((id, time.perf_counter()))


def perf_end(id, quiet=False):
    end = time.perf_counter()
    # find the most recent open marker with this id (search from the top of the stack)
    for i in range(len(_open) - 1, -1, -1):
        if _open[i][0] == id:
            break
    else:
        raise ValueError(f"perf_end({id!r}) without a matching perf_start")
    _, start = _open.pop(i)
    elapsed = end - start

    entry = _totals.setdefault(id, [0.0, 0])
    entry[0] += elapsed
    entry[1] += 1

    if not quiet:
        depth = i   # how many markers were still open around this one
        print(f"{'  ' * depth}{id}: {elapsed * 1000:.3f} ms")
    return elapsed


def perf_report():
    if not _totals:
        print("perf_report: nothing measured")
        return
    print(f"{'id':<20} {'total ms':>12} {'count':>8} {'avg ms':>10}")
    for id, (total, count) in sorted(_totals.items(), key=lambda kv: -kv[1][0]):
        print(f"{id:<20} {total * 1000:>12.3f} {count:>8} {total / count * 1000:>10.3f}")


def perf_reset():
    _open.clear()
    _totals.clear()
