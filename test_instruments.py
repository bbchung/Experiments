"""Stock, single-stock futures and warrant relationships from contract files."""

import tempfile
import unittest
from pathlib import Path

from AstraResearch.Experiments.instruments import NameIndex, stock_futures, warrants
from AstraResearch.io import ContractError

HEADER = "symbol,exchange,name,unit,limit_up,limit_down,day_trade,state,underlying,maturity_date\n"


class InstrumentMapTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        files = {
            "tse/stock": "2330,TSE,台積電,1000,1,1,Yes,n,,\n3711,TSE,日月光投控,1000,1,1,Yes,n,,\n2327,TSE,國巨*,1000,1,1,Yes,n,,\n0050,TSE,元大台灣50,1000,1,1,Yes,n,,\n"
            "00631L,TSE,元大台灣50正2,1000,1,1,Yes,n,,\n065466,TSE,台積電富邦5A購01,1000,1,0,No,n,台積電,20261016\n",
            "otc/stock": "6415,OTC,矽力*-KY,1000,1,1,Yes,n,,\n25283,OTC,皇普三,1000,1,1,No,n,,\n",
            "tse/warrant": "065466,TSE,台積電富邦5A購01,1000,1,0,No,n,台積電,20261016\n03164T,TSE,日月光中信5B售01,1000,1,0,No,n,日月光,20261016\n"
            "03004T,TSE,臺股指群益5C售05,1000,1,0,No,n,臺股指,20261216\n04001T,TSE,T50正2元大5A購01,1000,1,0,No,n,T50正2,20261216\n",
            "otc/warrant": "700008,OTC,矽力元大5A購01,1000,1,0,No,n,矽力,20261016\n",
            "taifex/futures": "CDFI6,TAIFEX,台積電期貨09,1,2715,2224,Yes,n,,\nCDFJ6,TAIFEX,台積電期貨10,1,2725,2233,Yes,n,,\n"
            "QFFI6,TAIFEX,小型台積電期貨09,1,2715,2224,Yes,n,,\nTXFI6,TAIFEX,臺股期貨09,1,1,1,Yes,n,,\nEXFI6,TAIFEX,電子期貨09,1,1,1,Yes,n,,\n",
        }
        for directory, rows in files.items():
            (self.root / directory).mkdir(parents=True)
            (self.root / directory / "20260910.csv").write_text(HEADER + rows)

    def test_stock_futures_carry_their_underlying_and_contract_size(self):
        rows = {row["product"]: row for row in stock_futures(self.root, "20260910")}
        self.assertEqual(set(rows), {"CDF", "QFF"})
        self.assertEqual((rows["CDF"]["underlying"], rows["CDF"]["shares_per_contract"], rows["CDF"]["contracts"]), ("2330", 2000, "CDFI6 CDFJ6"))
        self.assertEqual((rows["QFF"]["underlying"], rows["QFF"]["size"], rows["QFF"]["shares_per_contract"]), ("2330", "small", 100))
        self.assertEqual(rows["CDF"]["rule"], "legacy_name_exact")

    def test_authoritative_symbol_maps_unknown_name_and_keeps_six_character_ids(self):
        path = self.root / "taifex/futures/20260910.csv"
        path.write_text(
            HEADER.rstrip() + ",underlying_symbol\n"
            "CDFI6,TAIFEX,Updated official display name,1,1,1,Yes,n,,,2330\n"
            "ABFI6,TAIFEX,New ETF name,1,1,1,Yes,n,,,00631L\n"
            "TXFI6,TAIFEX,臺股期貨09,1,1,1,Yes,n,,,\n"
        )
        rows = {row["product"]: row for row in stock_futures(self.root, "20260910")}
        self.assertEqual(set(rows), {"CDF", "ABF"})
        self.assertEqual((rows["CDF"]["underlying"], rows["CDF"]["rule"]), ("2330", "underlying_symbol"))
        self.assertEqual(rows["ABF"]["underlying"], "00631L")
        self.assertIsNone(rows["CDF"]["shares_per_contract"])

    def test_authoritative_name_conflict_fails_instead_of_silently_remapping(self):
        path = self.root / "taifex/futures/20260910.csv"
        path.write_text(HEADER.rstrip() + ",underlying_symbol\nCDFI6,TAIFEX,台積電期貨09,1,1,1,Yes,n,,,3711\n")
        with self.assertRaisesRegex(ContractError, "conflicts with name mapping"):
            stock_futures(self.root, "20260910")

    def test_authoritative_symbol_requires_same_day_listing_and_mixed_evidence_is_visible(self):
        path = self.root / "taifex/futures/20260910.csv"
        header = HEADER.rstrip() + ",underlying_symbol\n"
        path.write_text(header + "CDFI6,TAIFEX,台積電期貨09,1,1,1,Yes,n,,,9999\n")
        with self.assertRaisesRegex(ContractError, "not listed"):
            stock_futures(self.root, "20260910")
        path.write_text(header + "CDFI6,TAIFEX,台積電期貨09,1,1,1,Yes,n,,,2330\nCDFJ6,TAIFEX,台積電期貨10,1,1,1,Yes,n,,,\n")
        self.assertEqual(stock_futures(self.root, "20260910")[0]["rule"], "mixed_authority_legacy")

    def test_mixed_legacy_names_do_not_imply_authoritative_identity(self):
        path = self.root / "taifex/futures/20260910.csv"
        path.write_text(HEADER + "GJFI6,TAIFEX,國巨*期貨09,1,1,1,Yes,n,,\nGJFJ6,TAIFEX,國巨期貨10,1,1,1,Yes,n,,\n")
        row = stock_futures(self.root, "20260910")[0]
        self.assertEqual((row["underlying"], row["rule"]), ("2327", "legacy_name_mixed"))

    def test_warrant_underlyings_resolve_or_stay_explicitly_unmapped(self):
        rows = {row["symbol"]: row for row in warrants(self.root, "20260910")}
        self.assertEqual((rows["065466"]["underlying"], rows["065466"]["side"], rows["065466"]["rule"]), ("2330", "call", "exact"))
        self.assertEqual((rows["03164T"]["underlying"], rows["03164T"]["side"], rows["03164T"]["rule"]), ("3711", "put", "unique_prefix"))
        self.assertEqual((rows["03004T"]["underlying"], rows["03004T"]["rule"]), ("TAIEX", "index"))
        self.assertEqual((rows["700008"]["underlying"], rows["700008"]["rule"]), ("6415", "normalized"))
        # An abbreviation shared by several funds is not guessed.
        self.assertEqual((rows["04001T"]["underlying"], rows["04001T"]["rule"]), (None, "unmapped"))

    def test_prefix_matches_must_be_unique_common_stocks(self):
        index = NameIndex({"國巨*": "2327", "國巨電子": "9999", "元大台灣50": "0050", "元大台灣50正2": "00631L"})
        self.assertEqual(index.resolve("國巨"), ("2327", "normalized"))
        self.assertEqual(index.resolve("元大台灣"), (None, "unmapped"))
        self.assertEqual(index.resolve(""), (None, "unmapped"))


if __name__ == "__main__":
    unittest.main()
