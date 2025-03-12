import os
import sys


def import_rttm(file, sep=None):
    if sep is None:
        with open(file, 'r') as f:
            first_line = f.readline()
            if "\t" in first_line:
                sep = "\t"
            else:
                sep = " "
        return import_rttm(file, sep)
    with open(file, 'r') as f:
        return [import_rttm_line(l, sep) for l in f.readlines()]


def import_rttm_line(line, sep):
    fields = line.strip().split(sep)
    if sep == " ":
        fields = [sep.join(fields[:2]), *fields[2:]]
    assert len(fields) in [8, 9], "Invalid line: {}".format(line)
    ID = fields[:2]
    start = float(fields[2])
    duration = float(fields[3])
    spk = fields[6]
    EXTRA = (fields[4:6], fields[7:])
    return [ID, start, duration, spk, EXTRA]


def export_rttm(file, data, sep=" ", check=False):
    last_start = 0
    with open(file, 'w') as f:
        for (ID, start, duration, spk, EXTRA) in data:
            if check:
                assert start >= last_start, f"Data not sorted! {start} < {last_start}"
            elif start < last_start:
                print(f"WARNING: Data not sorted! {start} < {last_start}")
            last_start = start + duration - 0.0111
            print(sep.join([*ID, str(start), str(duration),
                  *EXTRA[0], spk, *EXTRA[1]]), file=f)


def remove_overlaps(data, overlaps, eps=0.01, debug_verbose=False, print_with_meta=None):

    if overlaps is None:
        # We do not remove anything but make arbitrarily some uni-speaker segments

        new_data = []
        previous_end = 0
        found_overlap = False
        for (ID, start, duration, spk, EXTRA) in data:
            assert start >= 0, f"Invalid (negative) start: {start}"
            assert duration > 0, f"Invalid (non positive) duration: {duration}"
            end = start + duration
            if not found_overlap and start + eps < previous_end:
                found_overlap = True
                # Got an overlap
                previous_ID, previous_start, previous_duration, previous_spk, previous_EXTRA = new_data[-1]
                previous_end = previous_start + previous_duration
                if previous_end > end:
                    # the new segment is a small segment included in the previous one
                    if start > previous_start:
                        new_data[-1][2] = start - previous_start
                    else:
                        if start < previous_start - eps:
                            import pdb
                            pdb.set_trace()
                        assert start >= previous_start - eps
                        new_data.pop()
                    new_data.append([ID, start, duration, spk, EXTRA])
                    ID, spk, EXTRA = previous_ID, previous_spk, previous_EXTRA
                    start = end
                    duration = previous_end - start
                    assert duration > 0
                    new_data.append([ID, start, duration, spk, EXTRA])
                    end = previous_end
                else:
                    # The new segment starts before the other ends, and continue
                    new_start = 0.5 * (previous_end + start)
                    new_data[-1][2] = new_start - previous_start
                    new_data.append(
                        [ID, new_start, end - new_start, spk, EXTRA])
            else:
                new_data.append([ID, start, duration, spk, EXTRA])
            previous_end = end
        if found_overlap:
            return remove_overlaps(sorted(new_data, key=lambda x: x[1]), None, debug_verbose=debug_verbose, eps=eps)
        return new_data

    new_data = []
    i = 0  # index of overlap
    offset = 0  # second of signal removed
    previous_end = 0
    last_ID, last_start, last_duration, last_spk, last_EXTRA = None, None, None, None, None
    for (ID, start, duration, spk, EXTRA) in data:
        # Checking some assumptions on input data
        assert start + \
            eps >= previous_end, f"Data not sorted! {start} < {previous_end}"
        previous_end = start + duration
        assert duration > 0, f"Invalid (non positive) duration: {duration}"
        for start, duration, i, offset, offset_extra in process_overlap(overlaps, start, duration, i, offset, debug_verbose=debug_verbose, verbose_info=spk):
            if duration > 0:
                if last_ID == ID and last_spk == spk and last_EXTRA == EXTRA and last_start + last_duration >= start - eps:
                    new_data[-1] = [ID, last_start,
                                    last_duration + duration, spk, EXTRA]
                else:
                    new_data.append([ID, start, duration, spk, EXTRA])
                last_ID, last_start, last_duration, last_spk, last_EXTRA = new_data[-1]
                if debug_verbose:
                    print(new_data[-1][1:4])
            offset += offset_extra

    # Check output data
    for i in range(len(new_data) - 1):
        start, duration = new_data[i][1:3]
        next_start = new_data[i+1][1]
        assert start >= 0
        assert next_start >= 0
        assert next_start >= start + duration - \
            0.01, f"Data not sorted! {start} + {duration} = {start + duration - 0.01} > {next_start}"

    total_duration_in = sum([d[2] for d in data])
    total_duration_out = sum([d[2] for d in new_data])
    assert total_duration_out <= total_duration_in + eps
    if total_duration_out + eps < total_duration_in:
        total_duration_overlap = sum([end - start for start, end in overlaps])
        assert total_duration_overlap >= 0
        total_duration_removed = total_duration_in - total_duration_out
        if print_with_meta:
            print(f"Removed {total_duration_removed*100./total_duration_in:.2f}% ({total_duration_removed:.2f}/{total_duration_in:.2f} sec) of overlapping speech {print_with_meta}")
        assert total_duration_removed <= total_duration_overlap + 0.01

    return new_data


def process_overlap(overlaps, start, duration, i, offset, offset_extra=0, debug_verbose=False, verbose_info=""):
    # Only for debug debug_verbose
    duration_ref = duration
    offset_ref = offset
    offset_extra_ref = offset_extra
    start_overlap = -1
    end_overlap = -1
    end = start + duration

    def debug_info(print_output=True):
        s = f": {verbose_info} : {start:.2f}->{end:.2f} VS {start_overlap:.2f}->{end_overlap:.2f}"
        if duration > 0 and print_output:
            s += f" => duration:{duration_ref:.2f}->{duration:.2f}, offset:{offset_ref:.2f} + {offset-offset_ref:.2f} + {offset_extra:.2f} => {start-offset:.2f}-{duration:.2f}->{start+duration-offset:.2f}"
        else:
            s += f" => {' '*21}offset:{offset_ref:.2f} + {offset-offset_ref:.2f} + {offset_extra:.2f}"
        return s

    # If there are overlaps to process...
    if i < len(overlaps):
        start_overlap, end_overlap = overlaps[i]
        duration_overlap = end_overlap - start_overlap
        # If segment hits an overlap...
        if start <= end_overlap and end >= start_overlap:
            has_before = start <= start_overlap
            has_after = end >= end_overlap
            # If segment includes overlap...
            if has_before and has_after:
                duration = start_overlap - start
                offset_extra += duration_overlap
                i += 1
                if debug_verbose:
                    print("Segment includes overlap     " + debug_info())

                # The segment could include more overlap(s)
                return [(start - offset, duration, i, offset, offset_extra)] + \
                    process_overlap(overlaps, end_overlap, (end - end_overlap), i, offset +
                                    offset_extra, offset_extra=0, debug_verbose=debug_verbose, verbose_info=verbose_info)

            # If segment ends after overlap...
            elif has_after:
                duration = end - end_overlap
                eps = end_overlap - start
                assert eps >= 0
                offset += duration_overlap  # - eps
                # offset_extra += eps
                i += 1
                if debug_verbose:
                    print("Segment ends after overlap   " + debug_info(False))

                # The segment could include more overlap(s)
                return process_overlap(overlaps, end_overlap, duration, i, offset, offset_extra, debug_verbose=debug_verbose, verbose_info=verbose_info)

            # If segment starts before overlap...
            elif has_before:
                duration = start_overlap - start
                if debug_verbose:
                    print("Segment starts before overlap" + debug_info())

            # If segment is inside overlap...
            else:
                duration = 0
                if debug_verbose:
                    print("Segment is inside overlap    " + debug_info())

        else:
            # If we passed over some overlaps...
            if start >= end_overlap:
                offset += duration_overlap + offset_extra

                if debug_verbose:
                    print("We passed over some overlaps " + debug_info(False))
                return process_overlap(overlaps, start, duration, i+1, offset=offset, offset_extra=0, debug_verbose=debug_verbose, verbose_info=verbose_info)
            if debug_verbose:
                print("............................." + debug_info())
    elif debug_verbose:
        print("No more overlap              " + debug_info())

    return [(start - offset, duration, i, offset, offset_extra)]


def get_overlaps(data, func=lambda spk: " " in spk):
    overlaps = []
    previous_end = 0
    for (_, start, duration, spk, _) in data:
        if func(spk):
            overlaps.append((start, start + duration))
        elif start < previous_end:
            overlaps.append((start, previous_end))
        previous_end = start + duration
    return overlaps


def create_rttm_without_overlap(
    ref_file_in,
    hyp_file_in,
    ref_file_out,
    hyp_file_out,
    debug_verbose = False,
):
    ref = import_rttm(ref_file_in)
    hyp = import_rttm(hyp_file_in)

    overlaps = get_overlaps(ref)

    new_hyp = remove_overlaps(hyp, overlaps=None)
    if new_hyp != hyp:
        hyp = new_hyp
        print(f"Warning: overlaps found in (Hyp) {hyp_file_in}")
        # tmp_file = "_tmp".join(os.path.splitext(hyp_file_in))
        # print("Writing:", tmp_file)
        # export_rttm(tmp_file, new_hyp)

    new_hyp = remove_overlaps(hyp, overlaps, debug_verbose=debug_verbose, print_with_meta=f"(Hyp) {hyp_file_in}")
    new_ref = remove_overlaps(remove_overlaps(ref, overlaps=None), overlaps, debug_verbose=debug_verbose, print_with_meta=f"(Ref) {ref_file_in}")

    # print("Writing:", ref_file_out)
    export_rttm(ref_file_out, new_ref)
    # print("Writing:", hyp_file_out)
    export_rttm(hyp_file_out, new_hyp)

    return len(overlaps) > 0

if __name__ == "__main__":

    import argparse
    import glob

    parser = argparse.ArgumentParser(
        description='Remove overlaps from two RTTM file')
    parser.add_argument('ref_file', metavar='ref_file.rttm',
                        type=str, help='Reference RTTM file or folder')
    parser.add_argument('hyp_file', metavar='hyp_file.rttm',
                        type=str, help='Output RTTM file or folder')
    parser.add_argument('--debug_verbose', action='store_true', help='Debug debug_verbose')
    args = parser.parse_args()

    ref_files = args.ref_file
    hyp_files = args.hyp_file

    assert os.path.exists(
        ref_files), f"Reference file or folder does not exist: {ref_files}"
    assert os.path.exists(
        hyp_files), f"Output file or folder does not exist: {hyp_files}"
    assert os.path.isfile(hyp_files) == os.path.isfile(
        ref_files), "Reference and output must be both files or both folders"

    if os.path.isfile(ref_files):
        ref_files = [ref_files]
        hyp_files = [hyp_files]
    else:
        ref_files = sorted(glob.glob(os.path.join(ref_files, "*.rttm")))
        hyp_files = glob.glob(os.path.join(hyp_files, "*.rttm"))
        assert len(ref_files), "Could not find any reference rttm file"
        assert len(hyp_files), "Could not find any output rttm file"
        hyp_files = dict((os.path.basename(f), f) for f in hyp_files)
        ref_files, hyp_files = zip(*[(f, hyp_files[os.path.basename(f)])
                                   for f in ref_files if os.path.basename(f) in hyp_files and "nooverlap" not in f])
        assert len(
            ref_files), "Could not find any reference and output file with the same name"

    any_overlap = False

    for ref_file_in, hyp_file_in in zip(ref_files, hyp_files):
        if args.debug_verbose:
            print("Processing:", hyp_file_in, "VS", ref_file_in)

        ref_file_out = "_nooverlap".join(os.path.splitext(ref_file_in))
        hyp_file_out = "_nooverlap".join(os.path.splitext(hyp_file_in))

        has_overlap = \
        create_rttm_without_overlap(
            ref_file_in,
            hyp_file_in,
            ref_file_out,
            hyp_file_out,
            debug_verbose=args.debug_verbose,
        )

        any_overlap = any_overlap or has_overlap


    if not any_overlap:
        print("WARNING: No overlaps found in reference file(s) : make sure you did not confuse reference and output")
