"""Shared NooBook/NooBox builders for the noobox extract tests."""

from noobfriend.navigation import Footprint, NooBook, NooBox


def make_grism_book(
    *,
    ident: str = "00001",
    detector: str = "nrca1",
    pupil: str = "GRISMR",
    visit: str = "001",
    footprint: Footprint | None = None,
    shape: tuple[int, int] = (2048, 2048),
) -> NooBook:
    """Return a minimal 2bii WFSS NooBook keyed like a FRESCO exposure."""
    stem = f"jw01895001{visit}_02101_{ident}_{detector}_2bii"
    return NooBook(
        id=f"{stem}@2bii",
        location=f"/{stem}.fits",
        stage="2bii",
        program_id="01895",
        observation=("001",),
        visit=(visit,),
        ggsaa=("02101",),
        exposure=(ident,),
        detector=detector,
        pupil=pupil,
        filter="F444W",
        shape=shape,
        footprint=footprint,
    )


def make_box(*books: NooBook) -> NooBox:
    """Return a NooBox holding the given books."""
    box = NooBox()
    for book in books:
        box.add(book)
    return box
