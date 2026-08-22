"""E2: synthetic benchmark matrix — expected mode, face count and deviation
ceiling for a spread of CAD-export cases. Runs without the git-ignored
samples. Add a row here whenever a new geometry class is supported."""
import pytest
import cadquery as cq

from . import synth

CASES = [
    # name, builder, expected mode, expected ADVANCED_FACE count (None = don't check), max dev ceiling
    ('plate_holes',        synth.plate_holes,                    'prismatic', 10, 0.02),
    ('plate_holes_fillets', synth.plate_holes_fillets,           'prismatic', 14, 0.03),
    ('rounded_rect',       synth.rounded_rect,                   'prismatic', 10, 0.02),
    ('obround',            synth.obround,                        'prismatic', 10, 0.05),
    ('slot_hex',           synth.slot_hex,                       'prismatic', 17, 0.02),
    ('stepped_shaft',      synth.stepped_shaft,                  'prismatic', 7, 0.02),
    ('cross3',             synth.cross3,                         'prismatic', 30, 0.01),
    ('cross_blind',        synth.cross_blind,                    'prismatic', 9, 0.05),
    ('hollow_cube',        synth.hollow_cube,                    'prismatic', 12, 0.01),
    ('drafted_block',      synth.drafted_block,                  'prismatic', 6, 0.02),
    ('chamfer_top',        synth.chamfer_top,                    'prismatic', 10, 0.02),
    ('csk_plate',          synth.csk_plate,                      'prismatic', 14, 0.1),
    ('small_step_inside',  synth.small_step_inside,              'prismatic', 11, 0.02),
    ('tilted_3deg_x',      lambda: synth.rotate(synth.plate_holes(), (1, 0, 0), 3.0), 'prismatic', 10, 0.05),
    ('rotated_30_20',      lambda: synth.rotate(synth.plate_holes(), (0, 0, 1), 30).rotate((0, 0, 0), (1, 0, 0), 20), 'prismatic', 10, 0.02),
    ('tiny',               lambda: cq.Workplane('XY').box(3, 2, 0.6).faces('>Z').workplane().hole(0.5), 'prismatic', 7, 0.01),
    ('huge',               lambda: cq.Workplane('XY').box(1500, 900, 60).faces('>Z').workplane().hole(120), 'prismatic', 7, 0.1),
    ('L_bracket',          lambda: (cq.Workplane('XY').box(50, 30, 5).union(cq.Workplane('XY').box(5, 30, 40).translate((-22.5, 0, 17.5)))
                                    .faces('>Z').workplane().center(10, 0).hole(5)), 'prismatic', 9, 0.02),
    ('flat_gentle_arc',    synth.flat_gentle_arc,                'prismatic', 8, 0.02),    # flat run next to a gentle arc: the flat must stay a plane
    ('flat_gentle_arc_ledge', lambda: synth.flat_gentle_arc(0.5), 'prismatic', 13, 0.02),  # ... and across a 0.5 mm ledge, one cylinder, no slab seams
    ('thin_ledge_0p15',    lambda: synth.flat_gentle_arc(0.15),  'prismatic', 13, 0.02),   # a 0.15 mm ledge is a level of its own, not merged away
    ('sphere_boss',        synth.sphere_boss,                    'facegroup', 7, 0.05),    # face-group engine: 6 planes + 1 analytic sphere
    ('fillet_top',         synth.fillet_top,                     'facegroup', 10, 0.05),   # face-group engine: 6 planes + 4 cylinders
]


@pytest.mark.parametrize('name,builder,mode,faces,dev', CASES, ids=[c[0] for c in CASES])
def test_matrix(tmp_path, name, builder, mode, faces, dev):
    from stl2prism.pipeline import run
    p = synth.export(builder(), tmp_path / f'{name}.stl')
    out = str(tmp_path / f'{name}.step')
    r = run(p, out, verbose=False)
    assert r['mode'] == mode, (name, r['mode'], r['metrics'])
    got = synth.reimport(out)
    assert got['solids'] >= 1 and got['naked_edges'] == 0, (name, got)
    if faces is not None:
        assert got['faces'] == faces, (name, got['faces'])
    if dev is not None:
        assert r['metrics']['dev_max'] <= dev, (name, r['metrics']['dev_max'])
        if r['metrics'].get('vol_verified'):
            assert r['metrics']['vol_err_pct'] < 1.0
