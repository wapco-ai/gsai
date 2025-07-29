# Utility functions for route step display

from typing import List, Dict


def merge_short_segments(segments: List[Dict], threshold: float = 20.0) -> List[Dict]:
    """Merge short segments with adjacent ones for nicer directions display.

    Each segment is a dict with at least 'id', 'length', and 'angle' keys.
    The underlying path isn't modified; returned steps combine ids and lengths
    for display only.
    """
    if not segments:
        return []

    display_steps: List[Dict] = []
    i = 0
    # Copy segments to avoid mutating input
    segs = [dict(s) for s in segments]

    while i < len(segs):
        seg = segs[i]
        if seg['length'] >= threshold or len(segs) == 1:
            seg_ids = seg.get('segments', [seg['id']])
            display_steps.append({'segments': seg_ids,
                                  'length': seg['length'],
                                  'angle': seg['angle']})
            i += 1
            continue

        prev_diff = None
        next_diff = None
        if i > 0:
            prev_diff = abs(seg['angle'] - segs[i - 1]['angle'])
        if i < len(segs) - 1:
            next_diff = abs(seg['angle'] - segs[i + 1]['angle'])

        # Prefer merging with the neighbor that has the smaller angle difference
        if prev_diff is not None and (next_diff is None or prev_diff <= next_diff) and display_steps:
            # merge with previous step
            display_steps[-1]['segments'].append(seg['id'])
            display_steps[-1]['length'] += seg['length']
            i += 1
        else:
            # merge with next step
            if i + 1 < len(segs):
                segs[i + 1].setdefault('segments', [segs[i + 1]['id']])
                segs[i + 1]['segments'].insert(0, seg['id'])
                segs[i + 1]['length'] += seg['length']
            else:
                display_steps.append({'segments': [seg['id']],
                                      'length': seg['length'],
                                      'angle': seg['angle']})
            i += 1
    return display_steps


if __name__ == "__main__":
    example_segments = [
        {'id': 1, 'length': 20, 'angle': 0},
        {'id': 2, 'length': 9,  'angle': 10},
        {'id': 3, 'length': 30, 'angle': 12},
    ]
    merged = merge_short_segments(example_segments, threshold=20)
    for step in merged:
        print(step)
