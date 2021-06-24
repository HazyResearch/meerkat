import unittest

import numpy as np
import torch
from dosma.core.io.dicom_io import DicomReader
from pydicom.data import get_testdata_file

from meerkat.cells.volume import MedicalVolumeCell


class TestMedicalVolumeCell(unittest.TestCase):
    """Test the MedicalVolumeCell with sample slice data from pydicom."""

    # A slice of a CT - provided by pydicom
    _ct_file = get_testdata_file("CT_small.dcm")

    def test_basic_construction(self):
        cell = MedicalVolumeCell(self._ct_file, loader=DicomReader(group_by=None))
        assert isinstance(cell.loader, DicomReader)

        cell = MedicalVolumeCell(self._ct_file, loader=DicomReader(group_by=None))
        assert cell.transform is None
        assert isinstance(cell.loader, DicomReader)

    def test_transform(self):
        cell = MedicalVolumeCell(
            self._ct_file,
            loader=DicomReader(group_by=None),
            transform=lambda x: torch.from_numpy(np.asarray(x)),
        )
        out = cell.get()
        assert isinstance(out, torch.Tensor)

    def test_metadata(self):
        cell = MedicalVolumeCell(
            self._ct_file,
            loader=DicomReader(group_by=None),
        )
        _ = cell.get()
        assert cell.get_metadata() is None
        metadata = cell.get_metadata(force_load=True)
        assert metadata is not None
        assert "PixelData" not in metadata

        cell = MedicalVolumeCell(
            self._ct_file,
            loader=DicomReader(group_by=None),
            cache_metadata=True,
        )
        _ = cell.get()
        assert cell._metadata is not None
        metadata = cell._metadata
        assert "PixelData" not in metadata

        # Fetching the cell again should not update the underlying metadata.
        _ = cell.get()
        assert id(cell._metadata) == id(metadata)

        metadata = cell.get_metadata(ignore_bytes=True, readable=True, as_raw_type=True)
        assert all(not isinstance(v, bytes) for v in metadata.values())
        assert all(isinstance(k, str) for k in metadata)
        raw_types = (str, int, float, list, tuple)
        assert all(isinstance(v, raw_types) for v in metadata.values()), "\n".join(
            [
                f"{k} ({type(v)}): {v}"
                for k, v in metadata.items()
                if not isinstance(v, raw_types)
            ]
        )

    def test_state(self):
        dr = DicomReader(group_by=None)
        cell = MedicalVolumeCell(
            self._ct_file,
            loader=dr,
            transform=lambda x: torch.from_numpy(np.asarray(x)),
        )
        state = cell.get_state()

        cell2 = MedicalVolumeCell.from_state(state)

        assert cell.paths == cell2.paths
        assert isinstance(cell2.loader, DicomReader) and (cell2.loader.group_by is None)
        assert torch.all(cell.get() == cell2.get())
