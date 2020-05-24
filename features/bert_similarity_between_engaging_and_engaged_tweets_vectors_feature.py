from typing import List, Tuple

from google.cloud import bigquery, bigquery_storage_v1beta1
import pandas as pd

from base import BaseFeature, reduce_mem_usage


class BertSimilarityBetweenEngagingAndEngagedTweetsVectorsFeature(BaseFeature):
    # 使わない
    def import_columns(self) -> List[str]:
        ...
    def make_features(
        self, df_train_input: pd.DataFrame, df_test_input: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ...

    def read_and_save_features(
        self,
        train_table_name: str,
        test_table_name: str,
        train_output_path: str,
        test_output_path: str,
    ) -> None:
        df_train_features = self._read_from_bigquery(train_table_name)
        df_test_features = self._read_from_bigquery(test_table_name)
        df_train_features.columns = f"{self.name}_" + df_train_features.columns
        df_test_features.columns = f"{self.name}_" + df_test_features.columns

        if self.save_memory:
            self._logger.info("Reduce memory size - train data")
            df_train_features = reduce_mem_usage(df_train_features)
            self._logger.info("Reduce memory size - test data")
            df_test_features = reduce_mem_usage(df_test_features)

        self._logger.info(f"Saving features to {train_output_path}")
        df_train_features.to_feather(train_output_path)
        self._logger.info(f"Saving features to {test_output_path}")
        df_test_features.to_feather(test_output_path)

    def _read_from_bigquery(self, table_name: str) -> pd.DataFrame:
        self._logger.info(f"Reading from {table_name}")
        query = _QUERY.format(table_name=table_name)
        if self.debugging:
            query += " limit 10000"

        bqclient = bigquery.Client(project=self.PROJECT_ID)
        bqstorageclient = bigquery_storage_v1beta1.BigQueryStorageClient()
        df = (
            bqclient.query(query)
            .result()
            .to_dataframe(bqstorage_client=bqstorageclient)
        )
        return df

_QUERY = r"""
with unique_tweets as (
  select tweet_id, engaged_user_id
  from `recsys2020.training` t
  group by tweet_id, engaged_user_id
),
user_tweet_vectors as (
  select
    engaged_user_id as user_id,
    avg(gap_0) as gap_0,
    avg(gap_1) as gap_1,
    avg(gap_2) as gap_2,
    avg(gap_3) as gap_3,
    avg(gap_4) as gap_4,
    avg(gap_5) as gap_5,
    avg(gap_6) as gap_6,
    avg(gap_7) as gap_7,
    avg(gap_8) as gap_8,
    avg(gap_9) as gap_9,
    avg(gap_10) as gap_10,
    avg(gap_11) as gap_11,
    avg(gap_12) as gap_12,
    avg(gap_13) as gap_13,
    avg(gap_14) as gap_14,
    avg(gap_15) as gap_15,
    avg(gap_16) as gap_16,
    avg(gap_17) as gap_17,
    avg(gap_18) as gap_18,
    avg(gap_19) as gap_19,
    avg(gap_20) as gap_20,
    avg(gap_21) as gap_21,
    avg(gap_22) as gap_22,
    avg(gap_23) as gap_23,
    avg(gap_24) as gap_24,
    avg(gap_25) as gap_25,
    avg(gap_26) as gap_26,
    avg(gap_27) as gap_27,
    avg(gap_28) as gap_28,
    avg(gap_29) as gap_29,
    avg(gap_30) as gap_30,
    avg(gap_31) as gap_31,
    avg(gap_32) as gap_32,
    avg(gap_33) as gap_33,
    avg(gap_34) as gap_34,
    avg(gap_35) as gap_35,
    avg(gap_36) as gap_36,
    avg(gap_37) as gap_37,
    avg(gap_38) as gap_38,
    avg(gap_39) as gap_39,
    avg(gap_40) as gap_40,
    avg(gap_41) as gap_41,
    avg(gap_42) as gap_42,
    avg(gap_43) as gap_43,
    avg(gap_44) as gap_44,
    avg(gap_45) as gap_45,
    avg(gap_46) as gap_46,
    avg(gap_47) as gap_47,
    avg(gap_48) as gap_48,
    avg(gap_49) as gap_49,
    avg(gap_50) as gap_50,
    avg(gap_51) as gap_51,
    avg(gap_52) as gap_52,
    avg(gap_53) as gap_53,
    avg(gap_54) as gap_54,
    avg(gap_55) as gap_55,
    avg(gap_56) as gap_56,
    avg(gap_57) as gap_57,
    avg(gap_58) as gap_58,
    avg(gap_59) as gap_59,
    avg(gap_60) as gap_60,
    avg(gap_61) as gap_61,
    avg(gap_62) as gap_62,
    avg(gap_63) as gap_63,
    avg(gap_64) as gap_64,
    avg(gap_65) as gap_65,
    avg(gap_66) as gap_66,
    avg(gap_67) as gap_67,
    avg(gap_68) as gap_68,
    avg(gap_69) as gap_69,
    avg(gap_70) as gap_70,
    avg(gap_71) as gap_71,
    avg(gap_72) as gap_72,
    avg(gap_73) as gap_73,
    avg(gap_74) as gap_74,
    avg(gap_75) as gap_75,
    avg(gap_76) as gap_76,
    avg(gap_77) as gap_77,
    avg(gap_78) as gap_78,
    avg(gap_79) as gap_79,
    avg(gap_80) as gap_80,
    avg(gap_81) as gap_81,
    avg(gap_82) as gap_82,
    avg(gap_83) as gap_83,
    avg(gap_84) as gap_84,
    avg(gap_85) as gap_85,
    avg(gap_86) as gap_86,
    avg(gap_87) as gap_87,
    avg(gap_88) as gap_88,
    avg(gap_89) as gap_89,
    avg(gap_90) as gap_90,
    avg(gap_91) as gap_91,
    avg(gap_92) as gap_92,
    avg(gap_93) as gap_93,
    avg(gap_94) as gap_94,
    avg(gap_95) as gap_95,
    avg(gap_96) as gap_96,
    avg(gap_97) as gap_97,
    avg(gap_98) as gap_98,
    avg(gap_99) as gap_99,
    avg(gap_100) as gap_100,
    avg(gap_101) as gap_101,
    avg(gap_102) as gap_102,
    avg(gap_103) as gap_103,
    avg(gap_104) as gap_104,
    avg(gap_105) as gap_105,
    avg(gap_106) as gap_106,
    avg(gap_107) as gap_107,
    avg(gap_108) as gap_108,
    avg(gap_109) as gap_109,
    avg(gap_110) as gap_110,
    avg(gap_111) as gap_111,
    avg(gap_112) as gap_112,
    avg(gap_113) as gap_113,
    avg(gap_114) as gap_114,
    avg(gap_115) as gap_115,
    avg(gap_116) as gap_116,
    avg(gap_117) as gap_117,
    avg(gap_118) as gap_118,
    avg(gap_119) as gap_119,
    avg(gap_120) as gap_120,
    avg(gap_121) as gap_121,
    avg(gap_122) as gap_122,
    avg(gap_123) as gap_123,
    avg(gap_124) as gap_124,
    avg(gap_125) as gap_125,
    avg(gap_126) as gap_126,
    avg(gap_127) as gap_127,
    avg(gap_128) as gap_128,
    avg(gap_129) as gap_129,
    avg(gap_130) as gap_130,
    avg(gap_131) as gap_131,
    avg(gap_132) as gap_132,
    avg(gap_133) as gap_133,
    avg(gap_134) as gap_134,
    avg(gap_135) as gap_135,
    avg(gap_136) as gap_136,
    avg(gap_137) as gap_137,
    avg(gap_138) as gap_138,
    avg(gap_139) as gap_139,
    avg(gap_140) as gap_140,
    avg(gap_141) as gap_141,
    avg(gap_142) as gap_142,
    avg(gap_143) as gap_143,
    avg(gap_144) as gap_144,
    avg(gap_145) as gap_145,
    avg(gap_146) as gap_146,
    avg(gap_147) as gap_147,
    avg(gap_148) as gap_148,
    avg(gap_149) as gap_149,
    avg(gap_150) as gap_150,
    avg(gap_151) as gap_151,
    avg(gap_152) as gap_152,
    avg(gap_153) as gap_153,
    avg(gap_154) as gap_154,
    avg(gap_155) as gap_155,
    avg(gap_156) as gap_156,
    avg(gap_157) as gap_157,
    avg(gap_158) as gap_158,
    avg(gap_159) as gap_159,
    avg(gap_160) as gap_160,
    avg(gap_161) as gap_161,
    avg(gap_162) as gap_162,
    avg(gap_163) as gap_163,
    avg(gap_164) as gap_164,
    avg(gap_165) as gap_165,
    avg(gap_166) as gap_166,
    avg(gap_167) as gap_167,
    avg(gap_168) as gap_168,
    avg(gap_169) as gap_169,
    avg(gap_170) as gap_170,
    avg(gap_171) as gap_171,
    avg(gap_172) as gap_172,
    avg(gap_173) as gap_173,
    avg(gap_174) as gap_174,
    avg(gap_175) as gap_175,
    avg(gap_176) as gap_176,
    avg(gap_177) as gap_177,
    avg(gap_178) as gap_178,
    avg(gap_179) as gap_179,
    avg(gap_180) as gap_180,
    avg(gap_181) as gap_181,
    avg(gap_182) as gap_182,
    avg(gap_183) as gap_183,
    avg(gap_184) as gap_184,
    avg(gap_185) as gap_185,
    avg(gap_186) as gap_186,
    avg(gap_187) as gap_187,
    avg(gap_188) as gap_188,
    avg(gap_189) as gap_189,
    avg(gap_190) as gap_190,
    avg(gap_191) as gap_191,
    avg(gap_192) as gap_192,
    avg(gap_193) as gap_193,
    avg(gap_194) as gap_194,
    avg(gap_195) as gap_195,
    avg(gap_196) as gap_196,
    avg(gap_197) as gap_197,
    avg(gap_198) as gap_198,
    avg(gap_199) as gap_199,
    avg(gap_200) as gap_200,
    avg(gap_201) as gap_201,
    avg(gap_202) as gap_202,
    avg(gap_203) as gap_203,
    avg(gap_204) as gap_204,
    avg(gap_205) as gap_205,
    avg(gap_206) as gap_206,
    avg(gap_207) as gap_207,
    avg(gap_208) as gap_208,
    avg(gap_209) as gap_209,
    avg(gap_210) as gap_210,
    avg(gap_211) as gap_211,
    avg(gap_212) as gap_212,
    avg(gap_213) as gap_213,
    avg(gap_214) as gap_214,
    avg(gap_215) as gap_215,
    avg(gap_216) as gap_216,
    avg(gap_217) as gap_217,
    avg(gap_218) as gap_218,
    avg(gap_219) as gap_219,
    avg(gap_220) as gap_220,
    avg(gap_221) as gap_221,
    avg(gap_222) as gap_222,
    avg(gap_223) as gap_223,
    avg(gap_224) as gap_224,
    avg(gap_225) as gap_225,
    avg(gap_226) as gap_226,
    avg(gap_227) as gap_227,
    avg(gap_228) as gap_228,
    avg(gap_229) as gap_229,
    avg(gap_230) as gap_230,
    avg(gap_231) as gap_231,
    avg(gap_232) as gap_232,
    avg(gap_233) as gap_233,
    avg(gap_234) as gap_234,
    avg(gap_235) as gap_235,
    avg(gap_236) as gap_236,
    avg(gap_237) as gap_237,
    avg(gap_238) as gap_238,
    avg(gap_239) as gap_239,
    avg(gap_240) as gap_240,
    avg(gap_241) as gap_241,
    avg(gap_242) as gap_242,
    avg(gap_243) as gap_243,
    avg(gap_244) as gap_244,
    avg(gap_245) as gap_245,
    avg(gap_246) as gap_246,
    avg(gap_247) as gap_247,
    avg(gap_248) as gap_248,
    avg(gap_249) as gap_249,
    avg(gap_250) as gap_250,
    avg(gap_251) as gap_251,
    avg(gap_252) as gap_252,
    avg(gap_253) as gap_253,
    avg(gap_254) as gap_254,
    avg(gap_255) as gap_255,
    avg(gap_256) as gap_256,
    avg(gap_257) as gap_257,
    avg(gap_258) as gap_258,
    avg(gap_259) as gap_259,
    avg(gap_260) as gap_260,
    avg(gap_261) as gap_261,
    avg(gap_262) as gap_262,
    avg(gap_263) as gap_263,
    avg(gap_264) as gap_264,
    avg(gap_265) as gap_265,
    avg(gap_266) as gap_266,
    avg(gap_267) as gap_267,
    avg(gap_268) as gap_268,
    avg(gap_269) as gap_269,
    avg(gap_270) as gap_270,
    avg(gap_271) as gap_271,
    avg(gap_272) as gap_272,
    avg(gap_273) as gap_273,
    avg(gap_274) as gap_274,
    avg(gap_275) as gap_275,
    avg(gap_276) as gap_276,
    avg(gap_277) as gap_277,
    avg(gap_278) as gap_278,
    avg(gap_279) as gap_279,
    avg(gap_280) as gap_280,
    avg(gap_281) as gap_281,
    avg(gap_282) as gap_282,
    avg(gap_283) as gap_283,
    avg(gap_284) as gap_284,
    avg(gap_285) as gap_285,
    avg(gap_286) as gap_286,
    avg(gap_287) as gap_287,
    avg(gap_288) as gap_288,
    avg(gap_289) as gap_289,
    avg(gap_290) as gap_290,
    avg(gap_291) as gap_291,
    avg(gap_292) as gap_292,
    avg(gap_293) as gap_293,
    avg(gap_294) as gap_294,
    avg(gap_295) as gap_295,
    avg(gap_296) as gap_296,
    avg(gap_297) as gap_297,
    avg(gap_298) as gap_298,
    avg(gap_299) as gap_299,
    avg(gap_300) as gap_300,
    avg(gap_301) as gap_301,
    avg(gap_302) as gap_302,
    avg(gap_303) as gap_303,
    avg(gap_304) as gap_304,
    avg(gap_305) as gap_305,
    avg(gap_306) as gap_306,
    avg(gap_307) as gap_307,
    avg(gap_308) as gap_308,
    avg(gap_309) as gap_309,
    avg(gap_310) as gap_310,
    avg(gap_311) as gap_311,
    avg(gap_312) as gap_312,
    avg(gap_313) as gap_313,
    avg(gap_314) as gap_314,
    avg(gap_315) as gap_315,
    avg(gap_316) as gap_316,
    avg(gap_317) as gap_317,
    avg(gap_318) as gap_318,
    avg(gap_319) as gap_319,
    avg(gap_320) as gap_320,
    avg(gap_321) as gap_321,
    avg(gap_322) as gap_322,
    avg(gap_323) as gap_323,
    avg(gap_324) as gap_324,
    avg(gap_325) as gap_325,
    avg(gap_326) as gap_326,
    avg(gap_327) as gap_327,
    avg(gap_328) as gap_328,
    avg(gap_329) as gap_329,
    avg(gap_330) as gap_330,
    avg(gap_331) as gap_331,
    avg(gap_332) as gap_332,
    avg(gap_333) as gap_333,
    avg(gap_334) as gap_334,
    avg(gap_335) as gap_335,
    avg(gap_336) as gap_336,
    avg(gap_337) as gap_337,
    avg(gap_338) as gap_338,
    avg(gap_339) as gap_339,
    avg(gap_340) as gap_340,
    avg(gap_341) as gap_341,
    avg(gap_342) as gap_342,
    avg(gap_343) as gap_343,
    avg(gap_344) as gap_344,
    avg(gap_345) as gap_345,
    avg(gap_346) as gap_346,
    avg(gap_347) as gap_347,
    avg(gap_348) as gap_348,
    avg(gap_349) as gap_349,
    avg(gap_350) as gap_350,
    avg(gap_351) as gap_351,
    avg(gap_352) as gap_352,
    avg(gap_353) as gap_353,
    avg(gap_354) as gap_354,
    avg(gap_355) as gap_355,
    avg(gap_356) as gap_356,
    avg(gap_357) as gap_357,
    avg(gap_358) as gap_358,
    avg(gap_359) as gap_359,
    avg(gap_360) as gap_360,
    avg(gap_361) as gap_361,
    avg(gap_362) as gap_362,
    avg(gap_363) as gap_363,
    avg(gap_364) as gap_364,
    avg(gap_365) as gap_365,
    avg(gap_366) as gap_366,
    avg(gap_367) as gap_367,
    avg(gap_368) as gap_368,
    avg(gap_369) as gap_369,
    avg(gap_370) as gap_370,
    avg(gap_371) as gap_371,
    avg(gap_372) as gap_372,
    avg(gap_373) as gap_373,
    avg(gap_374) as gap_374,
    avg(gap_375) as gap_375,
    avg(gap_376) as gap_376,
    avg(gap_377) as gap_377,
    avg(gap_378) as gap_378,
    avg(gap_379) as gap_379,
    avg(gap_380) as gap_380,
    avg(gap_381) as gap_381,
    avg(gap_382) as gap_382,
    avg(gap_383) as gap_383,
    avg(gap_384) as gap_384,
    avg(gap_385) as gap_385,
    avg(gap_386) as gap_386,
    avg(gap_387) as gap_387,
    avg(gap_388) as gap_388,
    avg(gap_389) as gap_389,
    avg(gap_390) as gap_390,
    avg(gap_391) as gap_391,
    avg(gap_392) as gap_392,
    avg(gap_393) as gap_393,
    avg(gap_394) as gap_394,
    avg(gap_395) as gap_395,
    avg(gap_396) as gap_396,
    avg(gap_397) as gap_397,
    avg(gap_398) as gap_398,
    avg(gap_399) as gap_399,
    avg(gap_400) as gap_400,
    avg(gap_401) as gap_401,
    avg(gap_402) as gap_402,
    avg(gap_403) as gap_403,
    avg(gap_404) as gap_404,
    avg(gap_405) as gap_405,
    avg(gap_406) as gap_406,
    avg(gap_407) as gap_407,
    avg(gap_408) as gap_408,
    avg(gap_409) as gap_409,
    avg(gap_410) as gap_410,
    avg(gap_411) as gap_411,
    avg(gap_412) as gap_412,
    avg(gap_413) as gap_413,
    avg(gap_414) as gap_414,
    avg(gap_415) as gap_415,
    avg(gap_416) as gap_416,
    avg(gap_417) as gap_417,
    avg(gap_418) as gap_418,
    avg(gap_419) as gap_419,
    avg(gap_420) as gap_420,
    avg(gap_421) as gap_421,
    avg(gap_422) as gap_422,
    avg(gap_423) as gap_423,
    avg(gap_424) as gap_424,
    avg(gap_425) as gap_425,
    avg(gap_426) as gap_426,
    avg(gap_427) as gap_427,
    avg(gap_428) as gap_428,
    avg(gap_429) as gap_429,
    avg(gap_430) as gap_430,
    avg(gap_431) as gap_431,
    avg(gap_432) as gap_432,
    avg(gap_433) as gap_433,
    avg(gap_434) as gap_434,
    avg(gap_435) as gap_435,
    avg(gap_436) as gap_436,
    avg(gap_437) as gap_437,
    avg(gap_438) as gap_438,
    avg(gap_439) as gap_439,
    avg(gap_440) as gap_440,
    avg(gap_441) as gap_441,
    avg(gap_442) as gap_442,
    avg(gap_443) as gap_443,
    avg(gap_444) as gap_444,
    avg(gap_445) as gap_445,
    avg(gap_446) as gap_446,
    avg(gap_447) as gap_447,
    avg(gap_448) as gap_448,
    avg(gap_449) as gap_449,
    avg(gap_450) as gap_450,
    avg(gap_451) as gap_451,
    avg(gap_452) as gap_452,
    avg(gap_453) as gap_453,
    avg(gap_454) as gap_454,
    avg(gap_455) as gap_455,
    avg(gap_456) as gap_456,
    avg(gap_457) as gap_457,
    avg(gap_458) as gap_458,
    avg(gap_459) as gap_459,
    avg(gap_460) as gap_460,
    avg(gap_461) as gap_461,
    avg(gap_462) as gap_462,
    avg(gap_463) as gap_463,
    avg(gap_464) as gap_464,
    avg(gap_465) as gap_465,
    avg(gap_466) as gap_466,
    avg(gap_467) as gap_467,
    avg(gap_468) as gap_468,
    avg(gap_469) as gap_469,
    avg(gap_470) as gap_470,
    avg(gap_471) as gap_471,
    avg(gap_472) as gap_472,
    avg(gap_473) as gap_473,
    avg(gap_474) as gap_474,
    avg(gap_475) as gap_475,
    avg(gap_476) as gap_476,
    avg(gap_477) as gap_477,
    avg(gap_478) as gap_478,
    avg(gap_479) as gap_479,
    avg(gap_480) as gap_480,
    avg(gap_481) as gap_481,
    avg(gap_482) as gap_482,
    avg(gap_483) as gap_483,
    avg(gap_484) as gap_484,
    avg(gap_485) as gap_485,
    avg(gap_486) as gap_486,
    avg(gap_487) as gap_487,
    avg(gap_488) as gap_488,
    avg(gap_489) as gap_489,
    avg(gap_490) as gap_490,
    avg(gap_491) as gap_491,
    avg(gap_492) as gap_492,
    avg(gap_493) as gap_493,
    avg(gap_494) as gap_494,
    avg(gap_495) as gap_495,
    avg(gap_496) as gap_496,
    avg(gap_497) as gap_497,
    avg(gap_498) as gap_498,
    avg(gap_499) as gap_499,
    avg(gap_500) as gap_500,
    avg(gap_501) as gap_501,
    avg(gap_502) as gap_502,
    avg(gap_503) as gap_503,
    avg(gap_504) as gap_504,
    avg(gap_505) as gap_505,
    avg(gap_506) as gap_506,
    avg(gap_507) as gap_507,
    avg(gap_508) as gap_508,
    avg(gap_509) as gap_509,
    avg(gap_510) as gap_510,
    avg(gap_511) as gap_511,
    avg(gap_512) as gap_512,
    avg(gap_513) as gap_513,
    avg(gap_514) as gap_514,
    avg(gap_515) as gap_515,
    avg(gap_516) as gap_516,
    avg(gap_517) as gap_517,
    avg(gap_518) as gap_518,
    avg(gap_519) as gap_519,
    avg(gap_520) as gap_520,
    avg(gap_521) as gap_521,
    avg(gap_522) as gap_522,
    avg(gap_523) as gap_523,
    avg(gap_524) as gap_524,
    avg(gap_525) as gap_525,
    avg(gap_526) as gap_526,
    avg(gap_527) as gap_527,
    avg(gap_528) as gap_528,
    avg(gap_529) as gap_529,
    avg(gap_530) as gap_530,
    avg(gap_531) as gap_531,
    avg(gap_532) as gap_532,
    avg(gap_533) as gap_533,
    avg(gap_534) as gap_534,
    avg(gap_535) as gap_535,
    avg(gap_536) as gap_536,
    avg(gap_537) as gap_537,
    avg(gap_538) as gap_538,
    avg(gap_539) as gap_539,
    avg(gap_540) as gap_540,
    avg(gap_541) as gap_541,
    avg(gap_542) as gap_542,
    avg(gap_543) as gap_543,
    avg(gap_544) as gap_544,
    avg(gap_545) as gap_545,
    avg(gap_546) as gap_546,
    avg(gap_547) as gap_547,
    avg(gap_548) as gap_548,
    avg(gap_549) as gap_549,
    avg(gap_550) as gap_550,
    avg(gap_551) as gap_551,
    avg(gap_552) as gap_552,
    avg(gap_553) as gap_553,
    avg(gap_554) as gap_554,
    avg(gap_555) as gap_555,
    avg(gap_556) as gap_556,
    avg(gap_557) as gap_557,
    avg(gap_558) as gap_558,
    avg(gap_559) as gap_559,
    avg(gap_560) as gap_560,
    avg(gap_561) as gap_561,
    avg(gap_562) as gap_562,
    avg(gap_563) as gap_563,
    avg(gap_564) as gap_564,
    avg(gap_565) as gap_565,
    avg(gap_566) as gap_566,
    avg(gap_567) as gap_567,
    avg(gap_568) as gap_568,
    avg(gap_569) as gap_569,
    avg(gap_570) as gap_570,
    avg(gap_571) as gap_571,
    avg(gap_572) as gap_572,
    avg(gap_573) as gap_573,
    avg(gap_574) as gap_574,
    avg(gap_575) as gap_575,
    avg(gap_576) as gap_576,
    avg(gap_577) as gap_577,
    avg(gap_578) as gap_578,
    avg(gap_579) as gap_579,
    avg(gap_580) as gap_580,
    avg(gap_581) as gap_581,
    avg(gap_582) as gap_582,
    avg(gap_583) as gap_583,
    avg(gap_584) as gap_584,
    avg(gap_585) as gap_585,
    avg(gap_586) as gap_586,
    avg(gap_587) as gap_587,
    avg(gap_588) as gap_588,
    avg(gap_589) as gap_589,
    avg(gap_590) as gap_590,
    avg(gap_591) as gap_591,
    avg(gap_592) as gap_592,
    avg(gap_593) as gap_593,
    avg(gap_594) as gap_594,
    avg(gap_595) as gap_595,
    avg(gap_596) as gap_596,
    avg(gap_597) as gap_597,
    avg(gap_598) as gap_598,
    avg(gap_599) as gap_599,
    avg(gap_600) as gap_600,
    avg(gap_601) as gap_601,
    avg(gap_602) as gap_602,
    avg(gap_603) as gap_603,
    avg(gap_604) as gap_604,
    avg(gap_605) as gap_605,
    avg(gap_606) as gap_606,
    avg(gap_607) as gap_607,
    avg(gap_608) as gap_608,
    avg(gap_609) as gap_609,
    avg(gap_610) as gap_610,
    avg(gap_611) as gap_611,
    avg(gap_612) as gap_612,
    avg(gap_613) as gap_613,
    avg(gap_614) as gap_614,
    avg(gap_615) as gap_615,
    avg(gap_616) as gap_616,
    avg(gap_617) as gap_617,
    avg(gap_618) as gap_618,
    avg(gap_619) as gap_619,
    avg(gap_620) as gap_620,
    avg(gap_621) as gap_621,
    avg(gap_622) as gap_622,
    avg(gap_623) as gap_623,
    avg(gap_624) as gap_624,
    avg(gap_625) as gap_625,
    avg(gap_626) as gap_626,
    avg(gap_627) as gap_627,
    avg(gap_628) as gap_628,
    avg(gap_629) as gap_629,
    avg(gap_630) as gap_630,
    avg(gap_631) as gap_631,
    avg(gap_632) as gap_632,
    avg(gap_633) as gap_633,
    avg(gap_634) as gap_634,
    avg(gap_635) as gap_635,
    avg(gap_636) as gap_636,
    avg(gap_637) as gap_637,
    avg(gap_638) as gap_638,
    avg(gap_639) as gap_639,
    avg(gap_640) as gap_640,
    avg(gap_641) as gap_641,
    avg(gap_642) as gap_642,
    avg(gap_643) as gap_643,
    avg(gap_644) as gap_644,
    avg(gap_645) as gap_645,
    avg(gap_646) as gap_646,
    avg(gap_647) as gap_647,
    avg(gap_648) as gap_648,
    avg(gap_649) as gap_649,
    avg(gap_650) as gap_650,
    avg(gap_651) as gap_651,
    avg(gap_652) as gap_652,
    avg(gap_653) as gap_653,
    avg(gap_654) as gap_654,
    avg(gap_655) as gap_655,
    avg(gap_656) as gap_656,
    avg(gap_657) as gap_657,
    avg(gap_658) as gap_658,
    avg(gap_659) as gap_659,
    avg(gap_660) as gap_660,
    avg(gap_661) as gap_661,
    avg(gap_662) as gap_662,
    avg(gap_663) as gap_663,
    avg(gap_664) as gap_664,
    avg(gap_665) as gap_665,
    avg(gap_666) as gap_666,
    avg(gap_667) as gap_667,
    avg(gap_668) as gap_668,
    avg(gap_669) as gap_669,
    avg(gap_670) as gap_670,
    avg(gap_671) as gap_671,
    avg(gap_672) as gap_672,
    avg(gap_673) as gap_673,
    avg(gap_674) as gap_674,
    avg(gap_675) as gap_675,
    avg(gap_676) as gap_676,
    avg(gap_677) as gap_677,
    avg(gap_678) as gap_678,
    avg(gap_679) as gap_679,
    avg(gap_680) as gap_680,
    avg(gap_681) as gap_681,
    avg(gap_682) as gap_682,
    avg(gap_683) as gap_683,
    avg(gap_684) as gap_684,
    avg(gap_685) as gap_685,
    avg(gap_686) as gap_686,
    avg(gap_687) as gap_687,
    avg(gap_688) as gap_688,
    avg(gap_689) as gap_689,
    avg(gap_690) as gap_690,
    avg(gap_691) as gap_691,
    avg(gap_692) as gap_692,
    avg(gap_693) as gap_693,
    avg(gap_694) as gap_694,
    avg(gap_695) as gap_695,
    avg(gap_696) as gap_696,
    avg(gap_697) as gap_697,
    avg(gap_698) as gap_698,
    avg(gap_699) as gap_699,
    avg(gap_700) as gap_700,
    avg(gap_701) as gap_701,
    avg(gap_702) as gap_702,
    avg(gap_703) as gap_703,
    avg(gap_704) as gap_704,
    avg(gap_705) as gap_705,
    avg(gap_706) as gap_706,
    avg(gap_707) as gap_707,
    avg(gap_708) as gap_708,
    avg(gap_709) as gap_709,
    avg(gap_710) as gap_710,
    avg(gap_711) as gap_711,
    avg(gap_712) as gap_712,
    avg(gap_713) as gap_713,
    avg(gap_714) as gap_714,
    avg(gap_715) as gap_715,
    avg(gap_716) as gap_716,
    avg(gap_717) as gap_717,
    avg(gap_718) as gap_718,
    avg(gap_719) as gap_719,
    avg(gap_720) as gap_720,
    avg(gap_721) as gap_721,
    avg(gap_722) as gap_722,
    avg(gap_723) as gap_723,
    avg(gap_724) as gap_724,
    avg(gap_725) as gap_725,
    avg(gap_726) as gap_726,
    avg(gap_727) as gap_727,
    avg(gap_728) as gap_728,
    avg(gap_729) as gap_729,
    avg(gap_730) as gap_730,
    avg(gap_731) as gap_731,
    avg(gap_732) as gap_732,
    avg(gap_733) as gap_733,
    avg(gap_734) as gap_734,
    avg(gap_735) as gap_735,
    avg(gap_736) as gap_736,
    avg(gap_737) as gap_737,
    avg(gap_738) as gap_738,
    avg(gap_739) as gap_739,
    avg(gap_740) as gap_740,
    avg(gap_741) as gap_741,
    avg(gap_742) as gap_742,
    avg(gap_743) as gap_743,
    avg(gap_744) as gap_744,
    avg(gap_745) as gap_745,
    avg(gap_746) as gap_746,
    avg(gap_747) as gap_747,
    avg(gap_748) as gap_748,
    avg(gap_749) as gap_749,
    avg(gap_750) as gap_750,
    avg(gap_751) as gap_751,
    avg(gap_752) as gap_752,
    avg(gap_753) as gap_753,
    avg(gap_754) as gap_754,
    avg(gap_755) as gap_755,
    avg(gap_756) as gap_756,
    avg(gap_757) as gap_757,
    avg(gap_758) as gap_758,
    avg(gap_759) as gap_759,
    avg(gap_760) as gap_760,
    avg(gap_761) as gap_761,
    avg(gap_762) as gap_762,
    avg(gap_763) as gap_763,
    avg(gap_764) as gap_764,
    avg(gap_765) as gap_765,
    avg(gap_766) as gap_766,
    avg(gap_767) as gap_767
  from unique_tweets
  inner join `recsys2020.pretrained_bert_gap` gap on unique_tweets.tweet_id = gap.tweet_id
  group by user_id
)

select
  1.0 / 768 * (
    (engaging_gap.gap_0 * engaged_gap.gap_0) +
    (engaging_gap.gap_1 * engaged_gap.gap_1) +
    (engaging_gap.gap_2 * engaged_gap.gap_2) +
    (engaging_gap.gap_3 * engaged_gap.gap_3) +
    (engaging_gap.gap_4 * engaged_gap.gap_4) +
    (engaging_gap.gap_5 * engaged_gap.gap_5) +
    (engaging_gap.gap_6 * engaged_gap.gap_6) +
    (engaging_gap.gap_7 * engaged_gap.gap_7) +
    (engaging_gap.gap_8 * engaged_gap.gap_8) +
    (engaging_gap.gap_9 * engaged_gap.gap_9) +
    (engaging_gap.gap_10 * engaged_gap.gap_10) +
    (engaging_gap.gap_11 * engaged_gap.gap_11) +
    (engaging_gap.gap_12 * engaged_gap.gap_12) +
    (engaging_gap.gap_13 * engaged_gap.gap_13) +
    (engaging_gap.gap_14 * engaged_gap.gap_14) +
    (engaging_gap.gap_15 * engaged_gap.gap_15) +
    (engaging_gap.gap_16 * engaged_gap.gap_16) +
    (engaging_gap.gap_17 * engaged_gap.gap_17) +
    (engaging_gap.gap_18 * engaged_gap.gap_18) +
    (engaging_gap.gap_19 * engaged_gap.gap_19) +
    (engaging_gap.gap_20 * engaged_gap.gap_20) +
    (engaging_gap.gap_21 * engaged_gap.gap_21) +
    (engaging_gap.gap_22 * engaged_gap.gap_22) +
    (engaging_gap.gap_23 * engaged_gap.gap_23) +
    (engaging_gap.gap_24 * engaged_gap.gap_24) +
    (engaging_gap.gap_25 * engaged_gap.gap_25) +
    (engaging_gap.gap_26 * engaged_gap.gap_26) +
    (engaging_gap.gap_27 * engaged_gap.gap_27) +
    (engaging_gap.gap_28 * engaged_gap.gap_28) +
    (engaging_gap.gap_29 * engaged_gap.gap_29) +
    (engaging_gap.gap_30 * engaged_gap.gap_30) +
    (engaging_gap.gap_31 * engaged_gap.gap_31) +
    (engaging_gap.gap_32 * engaged_gap.gap_32) +
    (engaging_gap.gap_33 * engaged_gap.gap_33) +
    (engaging_gap.gap_34 * engaged_gap.gap_34) +
    (engaging_gap.gap_35 * engaged_gap.gap_35) +
    (engaging_gap.gap_36 * engaged_gap.gap_36) +
    (engaging_gap.gap_37 * engaged_gap.gap_37) +
    (engaging_gap.gap_38 * engaged_gap.gap_38) +
    (engaging_gap.gap_39 * engaged_gap.gap_39) +
    (engaging_gap.gap_40 * engaged_gap.gap_40) +
    (engaging_gap.gap_41 * engaged_gap.gap_41) +
    (engaging_gap.gap_42 * engaged_gap.gap_42) +
    (engaging_gap.gap_43 * engaged_gap.gap_43) +
    (engaging_gap.gap_44 * engaged_gap.gap_44) +
    (engaging_gap.gap_45 * engaged_gap.gap_45) +
    (engaging_gap.gap_46 * engaged_gap.gap_46) +
    (engaging_gap.gap_47 * engaged_gap.gap_47) +
    (engaging_gap.gap_48 * engaged_gap.gap_48) +
    (engaging_gap.gap_49 * engaged_gap.gap_49) +
    (engaging_gap.gap_50 * engaged_gap.gap_50) +
    (engaging_gap.gap_51 * engaged_gap.gap_51) +
    (engaging_gap.gap_52 * engaged_gap.gap_52) +
    (engaging_gap.gap_53 * engaged_gap.gap_53) +
    (engaging_gap.gap_54 * engaged_gap.gap_54) +
    (engaging_gap.gap_55 * engaged_gap.gap_55) +
    (engaging_gap.gap_56 * engaged_gap.gap_56) +
    (engaging_gap.gap_57 * engaged_gap.gap_57) +
    (engaging_gap.gap_58 * engaged_gap.gap_58) +
    (engaging_gap.gap_59 * engaged_gap.gap_59) +
    (engaging_gap.gap_60 * engaged_gap.gap_60) +
    (engaging_gap.gap_61 * engaged_gap.gap_61) +
    (engaging_gap.gap_62 * engaged_gap.gap_62) +
    (engaging_gap.gap_63 * engaged_gap.gap_63) +
    (engaging_gap.gap_64 * engaged_gap.gap_64) +
    (engaging_gap.gap_65 * engaged_gap.gap_65) +
    (engaging_gap.gap_66 * engaged_gap.gap_66) +
    (engaging_gap.gap_67 * engaged_gap.gap_67) +
    (engaging_gap.gap_68 * engaged_gap.gap_68) +
    (engaging_gap.gap_69 * engaged_gap.gap_69) +
    (engaging_gap.gap_70 * engaged_gap.gap_70) +
    (engaging_gap.gap_71 * engaged_gap.gap_71) +
    (engaging_gap.gap_72 * engaged_gap.gap_72) +
    (engaging_gap.gap_73 * engaged_gap.gap_73) +
    (engaging_gap.gap_74 * engaged_gap.gap_74) +
    (engaging_gap.gap_75 * engaged_gap.gap_75) +
    (engaging_gap.gap_76 * engaged_gap.gap_76) +
    (engaging_gap.gap_77 * engaged_gap.gap_77) +
    (engaging_gap.gap_78 * engaged_gap.gap_78) +
    (engaging_gap.gap_79 * engaged_gap.gap_79) +
    (engaging_gap.gap_80 * engaged_gap.gap_80) +
    (engaging_gap.gap_81 * engaged_gap.gap_81) +
    (engaging_gap.gap_82 * engaged_gap.gap_82) +
    (engaging_gap.gap_83 * engaged_gap.gap_83) +
    (engaging_gap.gap_84 * engaged_gap.gap_84) +
    (engaging_gap.gap_85 * engaged_gap.gap_85) +
    (engaging_gap.gap_86 * engaged_gap.gap_86) +
    (engaging_gap.gap_87 * engaged_gap.gap_87) +
    (engaging_gap.gap_88 * engaged_gap.gap_88) +
    (engaging_gap.gap_89 * engaged_gap.gap_89) +
    (engaging_gap.gap_90 * engaged_gap.gap_90) +
    (engaging_gap.gap_91 * engaged_gap.gap_91) +
    (engaging_gap.gap_92 * engaged_gap.gap_92) +
    (engaging_gap.gap_93 * engaged_gap.gap_93) +
    (engaging_gap.gap_94 * engaged_gap.gap_94) +
    (engaging_gap.gap_95 * engaged_gap.gap_95) +
    (engaging_gap.gap_96 * engaged_gap.gap_96) +
    (engaging_gap.gap_97 * engaged_gap.gap_97) +
    (engaging_gap.gap_98 * engaged_gap.gap_98) +
    (engaging_gap.gap_99 * engaged_gap.gap_99) +
    (engaging_gap.gap_100 * engaged_gap.gap_100) +
    (engaging_gap.gap_101 * engaged_gap.gap_101) +
    (engaging_gap.gap_102 * engaged_gap.gap_102) +
    (engaging_gap.gap_103 * engaged_gap.gap_103) +
    (engaging_gap.gap_104 * engaged_gap.gap_104) +
    (engaging_gap.gap_105 * engaged_gap.gap_105) +
    (engaging_gap.gap_106 * engaged_gap.gap_106) +
    (engaging_gap.gap_107 * engaged_gap.gap_107) +
    (engaging_gap.gap_108 * engaged_gap.gap_108) +
    (engaging_gap.gap_109 * engaged_gap.gap_109) +
    (engaging_gap.gap_110 * engaged_gap.gap_110) +
    (engaging_gap.gap_111 * engaged_gap.gap_111) +
    (engaging_gap.gap_112 * engaged_gap.gap_112) +
    (engaging_gap.gap_113 * engaged_gap.gap_113) +
    (engaging_gap.gap_114 * engaged_gap.gap_114) +
    (engaging_gap.gap_115 * engaged_gap.gap_115) +
    (engaging_gap.gap_116 * engaged_gap.gap_116) +
    (engaging_gap.gap_117 * engaged_gap.gap_117) +
    (engaging_gap.gap_118 * engaged_gap.gap_118) +
    (engaging_gap.gap_119 * engaged_gap.gap_119) +
    (engaging_gap.gap_120 * engaged_gap.gap_120) +
    (engaging_gap.gap_121 * engaged_gap.gap_121) +
    (engaging_gap.gap_122 * engaged_gap.gap_122) +
    (engaging_gap.gap_123 * engaged_gap.gap_123) +
    (engaging_gap.gap_124 * engaged_gap.gap_124) +
    (engaging_gap.gap_125 * engaged_gap.gap_125) +
    (engaging_gap.gap_126 * engaged_gap.gap_126) +
    (engaging_gap.gap_127 * engaged_gap.gap_127) +
    (engaging_gap.gap_128 * engaged_gap.gap_128) +
    (engaging_gap.gap_129 * engaged_gap.gap_129) +
    (engaging_gap.gap_130 * engaged_gap.gap_130) +
    (engaging_gap.gap_131 * engaged_gap.gap_131) +
    (engaging_gap.gap_132 * engaged_gap.gap_132) +
    (engaging_gap.gap_133 * engaged_gap.gap_133) +
    (engaging_gap.gap_134 * engaged_gap.gap_134) +
    (engaging_gap.gap_135 * engaged_gap.gap_135) +
    (engaging_gap.gap_136 * engaged_gap.gap_136) +
    (engaging_gap.gap_137 * engaged_gap.gap_137) +
    (engaging_gap.gap_138 * engaged_gap.gap_138) +
    (engaging_gap.gap_139 * engaged_gap.gap_139) +
    (engaging_gap.gap_140 * engaged_gap.gap_140) +
    (engaging_gap.gap_141 * engaged_gap.gap_141) +
    (engaging_gap.gap_142 * engaged_gap.gap_142) +
    (engaging_gap.gap_143 * engaged_gap.gap_143) +
    (engaging_gap.gap_144 * engaged_gap.gap_144) +
    (engaging_gap.gap_145 * engaged_gap.gap_145) +
    (engaging_gap.gap_146 * engaged_gap.gap_146) +
    (engaging_gap.gap_147 * engaged_gap.gap_147) +
    (engaging_gap.gap_148 * engaged_gap.gap_148) +
    (engaging_gap.gap_149 * engaged_gap.gap_149) +
    (engaging_gap.gap_150 * engaged_gap.gap_150) +
    (engaging_gap.gap_151 * engaged_gap.gap_151) +
    (engaging_gap.gap_152 * engaged_gap.gap_152) +
    (engaging_gap.gap_153 * engaged_gap.gap_153) +
    (engaging_gap.gap_154 * engaged_gap.gap_154) +
    (engaging_gap.gap_155 * engaged_gap.gap_155) +
    (engaging_gap.gap_156 * engaged_gap.gap_156) +
    (engaging_gap.gap_157 * engaged_gap.gap_157) +
    (engaging_gap.gap_158 * engaged_gap.gap_158) +
    (engaging_gap.gap_159 * engaged_gap.gap_159) +
    (engaging_gap.gap_160 * engaged_gap.gap_160) +
    (engaging_gap.gap_161 * engaged_gap.gap_161) +
    (engaging_gap.gap_162 * engaged_gap.gap_162) +
    (engaging_gap.gap_163 * engaged_gap.gap_163) +
    (engaging_gap.gap_164 * engaged_gap.gap_164) +
    (engaging_gap.gap_165 * engaged_gap.gap_165) +
    (engaging_gap.gap_166 * engaged_gap.gap_166) +
    (engaging_gap.gap_167 * engaged_gap.gap_167) +
    (engaging_gap.gap_168 * engaged_gap.gap_168) +
    (engaging_gap.gap_169 * engaged_gap.gap_169) +
    (engaging_gap.gap_170 * engaged_gap.gap_170) +
    (engaging_gap.gap_171 * engaged_gap.gap_171) +
    (engaging_gap.gap_172 * engaged_gap.gap_172) +
    (engaging_gap.gap_173 * engaged_gap.gap_173) +
    (engaging_gap.gap_174 * engaged_gap.gap_174) +
    (engaging_gap.gap_175 * engaged_gap.gap_175) +
    (engaging_gap.gap_176 * engaged_gap.gap_176) +
    (engaging_gap.gap_177 * engaged_gap.gap_177) +
    (engaging_gap.gap_178 * engaged_gap.gap_178) +
    (engaging_gap.gap_179 * engaged_gap.gap_179) +
    (engaging_gap.gap_180 * engaged_gap.gap_180) +
    (engaging_gap.gap_181 * engaged_gap.gap_181) +
    (engaging_gap.gap_182 * engaged_gap.gap_182) +
    (engaging_gap.gap_183 * engaged_gap.gap_183) +
    (engaging_gap.gap_184 * engaged_gap.gap_184) +
    (engaging_gap.gap_185 * engaged_gap.gap_185) +
    (engaging_gap.gap_186 * engaged_gap.gap_186) +
    (engaging_gap.gap_187 * engaged_gap.gap_187) +
    (engaging_gap.gap_188 * engaged_gap.gap_188) +
    (engaging_gap.gap_189 * engaged_gap.gap_189) +
    (engaging_gap.gap_190 * engaged_gap.gap_190) +
    (engaging_gap.gap_191 * engaged_gap.gap_191) +
    (engaging_gap.gap_192 * engaged_gap.gap_192) +
    (engaging_gap.gap_193 * engaged_gap.gap_193) +
    (engaging_gap.gap_194 * engaged_gap.gap_194) +
    (engaging_gap.gap_195 * engaged_gap.gap_195) +
    (engaging_gap.gap_196 * engaged_gap.gap_196) +
    (engaging_gap.gap_197 * engaged_gap.gap_197) +
    (engaging_gap.gap_198 * engaged_gap.gap_198) +
    (engaging_gap.gap_199 * engaged_gap.gap_199) +
    (engaging_gap.gap_200 * engaged_gap.gap_200) +
    (engaging_gap.gap_201 * engaged_gap.gap_201) +
    (engaging_gap.gap_202 * engaged_gap.gap_202) +
    (engaging_gap.gap_203 * engaged_gap.gap_203) +
    (engaging_gap.gap_204 * engaged_gap.gap_204) +
    (engaging_gap.gap_205 * engaged_gap.gap_205) +
    (engaging_gap.gap_206 * engaged_gap.gap_206) +
    (engaging_gap.gap_207 * engaged_gap.gap_207) +
    (engaging_gap.gap_208 * engaged_gap.gap_208) +
    (engaging_gap.gap_209 * engaged_gap.gap_209) +
    (engaging_gap.gap_210 * engaged_gap.gap_210) +
    (engaging_gap.gap_211 * engaged_gap.gap_211) +
    (engaging_gap.gap_212 * engaged_gap.gap_212) +
    (engaging_gap.gap_213 * engaged_gap.gap_213) +
    (engaging_gap.gap_214 * engaged_gap.gap_214) +
    (engaging_gap.gap_215 * engaged_gap.gap_215) +
    (engaging_gap.gap_216 * engaged_gap.gap_216) +
    (engaging_gap.gap_217 * engaged_gap.gap_217) +
    (engaging_gap.gap_218 * engaged_gap.gap_218) +
    (engaging_gap.gap_219 * engaged_gap.gap_219) +
    (engaging_gap.gap_220 * engaged_gap.gap_220) +
    (engaging_gap.gap_221 * engaged_gap.gap_221) +
    (engaging_gap.gap_222 * engaged_gap.gap_222) +
    (engaging_gap.gap_223 * engaged_gap.gap_223) +
    (engaging_gap.gap_224 * engaged_gap.gap_224) +
    (engaging_gap.gap_225 * engaged_gap.gap_225) +
    (engaging_gap.gap_226 * engaged_gap.gap_226) +
    (engaging_gap.gap_227 * engaged_gap.gap_227) +
    (engaging_gap.gap_228 * engaged_gap.gap_228) +
    (engaging_gap.gap_229 * engaged_gap.gap_229) +
    (engaging_gap.gap_230 * engaged_gap.gap_230) +
    (engaging_gap.gap_231 * engaged_gap.gap_231) +
    (engaging_gap.gap_232 * engaged_gap.gap_232) +
    (engaging_gap.gap_233 * engaged_gap.gap_233) +
    (engaging_gap.gap_234 * engaged_gap.gap_234) +
    (engaging_gap.gap_235 * engaged_gap.gap_235) +
    (engaging_gap.gap_236 * engaged_gap.gap_236) +
    (engaging_gap.gap_237 * engaged_gap.gap_237) +
    (engaging_gap.gap_238 * engaged_gap.gap_238) +
    (engaging_gap.gap_239 * engaged_gap.gap_239) +
    (engaging_gap.gap_240 * engaged_gap.gap_240) +
    (engaging_gap.gap_241 * engaged_gap.gap_241) +
    (engaging_gap.gap_242 * engaged_gap.gap_242) +
    (engaging_gap.gap_243 * engaged_gap.gap_243) +
    (engaging_gap.gap_244 * engaged_gap.gap_244) +
    (engaging_gap.gap_245 * engaged_gap.gap_245) +
    (engaging_gap.gap_246 * engaged_gap.gap_246) +
    (engaging_gap.gap_247 * engaged_gap.gap_247) +
    (engaging_gap.gap_248 * engaged_gap.gap_248) +
    (engaging_gap.gap_249 * engaged_gap.gap_249) +
    (engaging_gap.gap_250 * engaged_gap.gap_250) +
    (engaging_gap.gap_251 * engaged_gap.gap_251) +
    (engaging_gap.gap_252 * engaged_gap.gap_252) +
    (engaging_gap.gap_253 * engaged_gap.gap_253) +
    (engaging_gap.gap_254 * engaged_gap.gap_254) +
    (engaging_gap.gap_255 * engaged_gap.gap_255) +
    (engaging_gap.gap_256 * engaged_gap.gap_256) +
    (engaging_gap.gap_257 * engaged_gap.gap_257) +
    (engaging_gap.gap_258 * engaged_gap.gap_258) +
    (engaging_gap.gap_259 * engaged_gap.gap_259) +
    (engaging_gap.gap_260 * engaged_gap.gap_260) +
    (engaging_gap.gap_261 * engaged_gap.gap_261) +
    (engaging_gap.gap_262 * engaged_gap.gap_262) +
    (engaging_gap.gap_263 * engaged_gap.gap_263) +
    (engaging_gap.gap_264 * engaged_gap.gap_264) +
    (engaging_gap.gap_265 * engaged_gap.gap_265) +
    (engaging_gap.gap_266 * engaged_gap.gap_266) +
    (engaging_gap.gap_267 * engaged_gap.gap_267) +
    (engaging_gap.gap_268 * engaged_gap.gap_268) +
    (engaging_gap.gap_269 * engaged_gap.gap_269) +
    (engaging_gap.gap_270 * engaged_gap.gap_270) +
    (engaging_gap.gap_271 * engaged_gap.gap_271) +
    (engaging_gap.gap_272 * engaged_gap.gap_272) +
    (engaging_gap.gap_273 * engaged_gap.gap_273) +
    (engaging_gap.gap_274 * engaged_gap.gap_274) +
    (engaging_gap.gap_275 * engaged_gap.gap_275) +
    (engaging_gap.gap_276 * engaged_gap.gap_276) +
    (engaging_gap.gap_277 * engaged_gap.gap_277) +
    (engaging_gap.gap_278 * engaged_gap.gap_278) +
    (engaging_gap.gap_279 * engaged_gap.gap_279) +
    (engaging_gap.gap_280 * engaged_gap.gap_280) +
    (engaging_gap.gap_281 * engaged_gap.gap_281) +
    (engaging_gap.gap_282 * engaged_gap.gap_282) +
    (engaging_gap.gap_283 * engaged_gap.gap_283) +
    (engaging_gap.gap_284 * engaged_gap.gap_284) +
    (engaging_gap.gap_285 * engaged_gap.gap_285) +
    (engaging_gap.gap_286 * engaged_gap.gap_286) +
    (engaging_gap.gap_287 * engaged_gap.gap_287) +
    (engaging_gap.gap_288 * engaged_gap.gap_288) +
    (engaging_gap.gap_289 * engaged_gap.gap_289) +
    (engaging_gap.gap_290 * engaged_gap.gap_290) +
    (engaging_gap.gap_291 * engaged_gap.gap_291) +
    (engaging_gap.gap_292 * engaged_gap.gap_292) +
    (engaging_gap.gap_293 * engaged_gap.gap_293) +
    (engaging_gap.gap_294 * engaged_gap.gap_294) +
    (engaging_gap.gap_295 * engaged_gap.gap_295) +
    (engaging_gap.gap_296 * engaged_gap.gap_296) +
    (engaging_gap.gap_297 * engaged_gap.gap_297) +
    (engaging_gap.gap_298 * engaged_gap.gap_298) +
    (engaging_gap.gap_299 * engaged_gap.gap_299) +
    (engaging_gap.gap_300 * engaged_gap.gap_300) +
    (engaging_gap.gap_301 * engaged_gap.gap_301) +
    (engaging_gap.gap_302 * engaged_gap.gap_302) +
    (engaging_gap.gap_303 * engaged_gap.gap_303) +
    (engaging_gap.gap_304 * engaged_gap.gap_304) +
    (engaging_gap.gap_305 * engaged_gap.gap_305) +
    (engaging_gap.gap_306 * engaged_gap.gap_306) +
    (engaging_gap.gap_307 * engaged_gap.gap_307) +
    (engaging_gap.gap_308 * engaged_gap.gap_308) +
    (engaging_gap.gap_309 * engaged_gap.gap_309) +
    (engaging_gap.gap_310 * engaged_gap.gap_310) +
    (engaging_gap.gap_311 * engaged_gap.gap_311) +
    (engaging_gap.gap_312 * engaged_gap.gap_312) +
    (engaging_gap.gap_313 * engaged_gap.gap_313) +
    (engaging_gap.gap_314 * engaged_gap.gap_314) +
    (engaging_gap.gap_315 * engaged_gap.gap_315) +
    (engaging_gap.gap_316 * engaged_gap.gap_316) +
    (engaging_gap.gap_317 * engaged_gap.gap_317) +
    (engaging_gap.gap_318 * engaged_gap.gap_318) +
    (engaging_gap.gap_319 * engaged_gap.gap_319) +
    (engaging_gap.gap_320 * engaged_gap.gap_320) +
    (engaging_gap.gap_321 * engaged_gap.gap_321) +
    (engaging_gap.gap_322 * engaged_gap.gap_322) +
    (engaging_gap.gap_323 * engaged_gap.gap_323) +
    (engaging_gap.gap_324 * engaged_gap.gap_324) +
    (engaging_gap.gap_325 * engaged_gap.gap_325) +
    (engaging_gap.gap_326 * engaged_gap.gap_326) +
    (engaging_gap.gap_327 * engaged_gap.gap_327) +
    (engaging_gap.gap_328 * engaged_gap.gap_328) +
    (engaging_gap.gap_329 * engaged_gap.gap_329) +
    (engaging_gap.gap_330 * engaged_gap.gap_330) +
    (engaging_gap.gap_331 * engaged_gap.gap_331) +
    (engaging_gap.gap_332 * engaged_gap.gap_332) +
    (engaging_gap.gap_333 * engaged_gap.gap_333) +
    (engaging_gap.gap_334 * engaged_gap.gap_334) +
    (engaging_gap.gap_335 * engaged_gap.gap_335) +
    (engaging_gap.gap_336 * engaged_gap.gap_336) +
    (engaging_gap.gap_337 * engaged_gap.gap_337) +
    (engaging_gap.gap_338 * engaged_gap.gap_338) +
    (engaging_gap.gap_339 * engaged_gap.gap_339) +
    (engaging_gap.gap_340 * engaged_gap.gap_340) +
    (engaging_gap.gap_341 * engaged_gap.gap_341) +
    (engaging_gap.gap_342 * engaged_gap.gap_342) +
    (engaging_gap.gap_343 * engaged_gap.gap_343) +
    (engaging_gap.gap_344 * engaged_gap.gap_344) +
    (engaging_gap.gap_345 * engaged_gap.gap_345) +
    (engaging_gap.gap_346 * engaged_gap.gap_346) +
    (engaging_gap.gap_347 * engaged_gap.gap_347) +
    (engaging_gap.gap_348 * engaged_gap.gap_348) +
    (engaging_gap.gap_349 * engaged_gap.gap_349) +
    (engaging_gap.gap_350 * engaged_gap.gap_350) +
    (engaging_gap.gap_351 * engaged_gap.gap_351) +
    (engaging_gap.gap_352 * engaged_gap.gap_352) +
    (engaging_gap.gap_353 * engaged_gap.gap_353) +
    (engaging_gap.gap_354 * engaged_gap.gap_354) +
    (engaging_gap.gap_355 * engaged_gap.gap_355) +
    (engaging_gap.gap_356 * engaged_gap.gap_356) +
    (engaging_gap.gap_357 * engaged_gap.gap_357) +
    (engaging_gap.gap_358 * engaged_gap.gap_358) +
    (engaging_gap.gap_359 * engaged_gap.gap_359) +
    (engaging_gap.gap_360 * engaged_gap.gap_360) +
    (engaging_gap.gap_361 * engaged_gap.gap_361) +
    (engaging_gap.gap_362 * engaged_gap.gap_362) +
    (engaging_gap.gap_363 * engaged_gap.gap_363) +
    (engaging_gap.gap_364 * engaged_gap.gap_364) +
    (engaging_gap.gap_365 * engaged_gap.gap_365) +
    (engaging_gap.gap_366 * engaged_gap.gap_366) +
    (engaging_gap.gap_367 * engaged_gap.gap_367) +
    (engaging_gap.gap_368 * engaged_gap.gap_368) +
    (engaging_gap.gap_369 * engaged_gap.gap_369) +
    (engaging_gap.gap_370 * engaged_gap.gap_370) +
    (engaging_gap.gap_371 * engaged_gap.gap_371) +
    (engaging_gap.gap_372 * engaged_gap.gap_372) +
    (engaging_gap.gap_373 * engaged_gap.gap_373) +
    (engaging_gap.gap_374 * engaged_gap.gap_374) +
    (engaging_gap.gap_375 * engaged_gap.gap_375) +
    (engaging_gap.gap_376 * engaged_gap.gap_376) +
    (engaging_gap.gap_377 * engaged_gap.gap_377) +
    (engaging_gap.gap_378 * engaged_gap.gap_378) +
    (engaging_gap.gap_379 * engaged_gap.gap_379) +
    (engaging_gap.gap_380 * engaged_gap.gap_380) +
    (engaging_gap.gap_381 * engaged_gap.gap_381) +
    (engaging_gap.gap_382 * engaged_gap.gap_382) +
    (engaging_gap.gap_383 * engaged_gap.gap_383) +
    (engaging_gap.gap_384 * engaged_gap.gap_384) +
    (engaging_gap.gap_385 * engaged_gap.gap_385) +
    (engaging_gap.gap_386 * engaged_gap.gap_386) +
    (engaging_gap.gap_387 * engaged_gap.gap_387) +
    (engaging_gap.gap_388 * engaged_gap.gap_388) +
    (engaging_gap.gap_389 * engaged_gap.gap_389) +
    (engaging_gap.gap_390 * engaged_gap.gap_390) +
    (engaging_gap.gap_391 * engaged_gap.gap_391) +
    (engaging_gap.gap_392 * engaged_gap.gap_392) +
    (engaging_gap.gap_393 * engaged_gap.gap_393) +
    (engaging_gap.gap_394 * engaged_gap.gap_394) +
    (engaging_gap.gap_395 * engaged_gap.gap_395) +
    (engaging_gap.gap_396 * engaged_gap.gap_396) +
    (engaging_gap.gap_397 * engaged_gap.gap_397) +
    (engaging_gap.gap_398 * engaged_gap.gap_398) +
    (engaging_gap.gap_399 * engaged_gap.gap_399) +
    (engaging_gap.gap_400 * engaged_gap.gap_400) +
    (engaging_gap.gap_401 * engaged_gap.gap_401) +
    (engaging_gap.gap_402 * engaged_gap.gap_402) +
    (engaging_gap.gap_403 * engaged_gap.gap_403) +
    (engaging_gap.gap_404 * engaged_gap.gap_404) +
    (engaging_gap.gap_405 * engaged_gap.gap_405) +
    (engaging_gap.gap_406 * engaged_gap.gap_406) +
    (engaging_gap.gap_407 * engaged_gap.gap_407) +
    (engaging_gap.gap_408 * engaged_gap.gap_408) +
    (engaging_gap.gap_409 * engaged_gap.gap_409) +
    (engaging_gap.gap_410 * engaged_gap.gap_410) +
    (engaging_gap.gap_411 * engaged_gap.gap_411) +
    (engaging_gap.gap_412 * engaged_gap.gap_412) +
    (engaging_gap.gap_413 * engaged_gap.gap_413) +
    (engaging_gap.gap_414 * engaged_gap.gap_414) +
    (engaging_gap.gap_415 * engaged_gap.gap_415) +
    (engaging_gap.gap_416 * engaged_gap.gap_416) +
    (engaging_gap.gap_417 * engaged_gap.gap_417) +
    (engaging_gap.gap_418 * engaged_gap.gap_418) +
    (engaging_gap.gap_419 * engaged_gap.gap_419) +
    (engaging_gap.gap_420 * engaged_gap.gap_420) +
    (engaging_gap.gap_421 * engaged_gap.gap_421) +
    (engaging_gap.gap_422 * engaged_gap.gap_422) +
    (engaging_gap.gap_423 * engaged_gap.gap_423) +
    (engaging_gap.gap_424 * engaged_gap.gap_424) +
    (engaging_gap.gap_425 * engaged_gap.gap_425) +
    (engaging_gap.gap_426 * engaged_gap.gap_426) +
    (engaging_gap.gap_427 * engaged_gap.gap_427) +
    (engaging_gap.gap_428 * engaged_gap.gap_428) +
    (engaging_gap.gap_429 * engaged_gap.gap_429) +
    (engaging_gap.gap_430 * engaged_gap.gap_430) +
    (engaging_gap.gap_431 * engaged_gap.gap_431) +
    (engaging_gap.gap_432 * engaged_gap.gap_432) +
    (engaging_gap.gap_433 * engaged_gap.gap_433) +
    (engaging_gap.gap_434 * engaged_gap.gap_434) +
    (engaging_gap.gap_435 * engaged_gap.gap_435) +
    (engaging_gap.gap_436 * engaged_gap.gap_436) +
    (engaging_gap.gap_437 * engaged_gap.gap_437) +
    (engaging_gap.gap_438 * engaged_gap.gap_438) +
    (engaging_gap.gap_439 * engaged_gap.gap_439) +
    (engaging_gap.gap_440 * engaged_gap.gap_440) +
    (engaging_gap.gap_441 * engaged_gap.gap_441) +
    (engaging_gap.gap_442 * engaged_gap.gap_442) +
    (engaging_gap.gap_443 * engaged_gap.gap_443) +
    (engaging_gap.gap_444 * engaged_gap.gap_444) +
    (engaging_gap.gap_445 * engaged_gap.gap_445) +
    (engaging_gap.gap_446 * engaged_gap.gap_446) +
    (engaging_gap.gap_447 * engaged_gap.gap_447) +
    (engaging_gap.gap_448 * engaged_gap.gap_448) +
    (engaging_gap.gap_449 * engaged_gap.gap_449) +
    (engaging_gap.gap_450 * engaged_gap.gap_450) +
    (engaging_gap.gap_451 * engaged_gap.gap_451) +
    (engaging_gap.gap_452 * engaged_gap.gap_452) +
    (engaging_gap.gap_453 * engaged_gap.gap_453) +
    (engaging_gap.gap_454 * engaged_gap.gap_454) +
    (engaging_gap.gap_455 * engaged_gap.gap_455) +
    (engaging_gap.gap_456 * engaged_gap.gap_456) +
    (engaging_gap.gap_457 * engaged_gap.gap_457) +
    (engaging_gap.gap_458 * engaged_gap.gap_458) +
    (engaging_gap.gap_459 * engaged_gap.gap_459) +
    (engaging_gap.gap_460 * engaged_gap.gap_460) +
    (engaging_gap.gap_461 * engaged_gap.gap_461) +
    (engaging_gap.gap_462 * engaged_gap.gap_462) +
    (engaging_gap.gap_463 * engaged_gap.gap_463) +
    (engaging_gap.gap_464 * engaged_gap.gap_464) +
    (engaging_gap.gap_465 * engaged_gap.gap_465) +
    (engaging_gap.gap_466 * engaged_gap.gap_466) +
    (engaging_gap.gap_467 * engaged_gap.gap_467) +
    (engaging_gap.gap_468 * engaged_gap.gap_468) +
    (engaging_gap.gap_469 * engaged_gap.gap_469) +
    (engaging_gap.gap_470 * engaged_gap.gap_470) +
    (engaging_gap.gap_471 * engaged_gap.gap_471) +
    (engaging_gap.gap_472 * engaged_gap.gap_472) +
    (engaging_gap.gap_473 * engaged_gap.gap_473) +
    (engaging_gap.gap_474 * engaged_gap.gap_474) +
    (engaging_gap.gap_475 * engaged_gap.gap_475) +
    (engaging_gap.gap_476 * engaged_gap.gap_476) +
    (engaging_gap.gap_477 * engaged_gap.gap_477) +
    (engaging_gap.gap_478 * engaged_gap.gap_478) +
    (engaging_gap.gap_479 * engaged_gap.gap_479) +
    (engaging_gap.gap_480 * engaged_gap.gap_480) +
    (engaging_gap.gap_481 * engaged_gap.gap_481) +
    (engaging_gap.gap_482 * engaged_gap.gap_482) +
    (engaging_gap.gap_483 * engaged_gap.gap_483) +
    (engaging_gap.gap_484 * engaged_gap.gap_484) +
    (engaging_gap.gap_485 * engaged_gap.gap_485) +
    (engaging_gap.gap_486 * engaged_gap.gap_486) +
    (engaging_gap.gap_487 * engaged_gap.gap_487) +
    (engaging_gap.gap_488 * engaged_gap.gap_488) +
    (engaging_gap.gap_489 * engaged_gap.gap_489) +
    (engaging_gap.gap_490 * engaged_gap.gap_490) +
    (engaging_gap.gap_491 * engaged_gap.gap_491) +
    (engaging_gap.gap_492 * engaged_gap.gap_492) +
    (engaging_gap.gap_493 * engaged_gap.gap_493) +
    (engaging_gap.gap_494 * engaged_gap.gap_494) +
    (engaging_gap.gap_495 * engaged_gap.gap_495) +
    (engaging_gap.gap_496 * engaged_gap.gap_496) +
    (engaging_gap.gap_497 * engaged_gap.gap_497) +
    (engaging_gap.gap_498 * engaged_gap.gap_498) +
    (engaging_gap.gap_499 * engaged_gap.gap_499) +
    (engaging_gap.gap_500 * engaged_gap.gap_500) +
    (engaging_gap.gap_501 * engaged_gap.gap_501) +
    (engaging_gap.gap_502 * engaged_gap.gap_502) +
    (engaging_gap.gap_503 * engaged_gap.gap_503) +
    (engaging_gap.gap_504 * engaged_gap.gap_504) +
    (engaging_gap.gap_505 * engaged_gap.gap_505) +
    (engaging_gap.gap_506 * engaged_gap.gap_506) +
    (engaging_gap.gap_507 * engaged_gap.gap_507) +
    (engaging_gap.gap_508 * engaged_gap.gap_508) +
    (engaging_gap.gap_509 * engaged_gap.gap_509) +
    (engaging_gap.gap_510 * engaged_gap.gap_510) +
    (engaging_gap.gap_511 * engaged_gap.gap_511) +
    (engaging_gap.gap_512 * engaged_gap.gap_512) +
    (engaging_gap.gap_513 * engaged_gap.gap_513) +
    (engaging_gap.gap_514 * engaged_gap.gap_514) +
    (engaging_gap.gap_515 * engaged_gap.gap_515) +
    (engaging_gap.gap_516 * engaged_gap.gap_516) +
    (engaging_gap.gap_517 * engaged_gap.gap_517) +
    (engaging_gap.gap_518 * engaged_gap.gap_518) +
    (engaging_gap.gap_519 * engaged_gap.gap_519) +
    (engaging_gap.gap_520 * engaged_gap.gap_520) +
    (engaging_gap.gap_521 * engaged_gap.gap_521) +
    (engaging_gap.gap_522 * engaged_gap.gap_522) +
    (engaging_gap.gap_523 * engaged_gap.gap_523) +
    (engaging_gap.gap_524 * engaged_gap.gap_524) +
    (engaging_gap.gap_525 * engaged_gap.gap_525) +
    (engaging_gap.gap_526 * engaged_gap.gap_526) +
    (engaging_gap.gap_527 * engaged_gap.gap_527) +
    (engaging_gap.gap_528 * engaged_gap.gap_528) +
    (engaging_gap.gap_529 * engaged_gap.gap_529) +
    (engaging_gap.gap_530 * engaged_gap.gap_530) +
    (engaging_gap.gap_531 * engaged_gap.gap_531) +
    (engaging_gap.gap_532 * engaged_gap.gap_532) +
    (engaging_gap.gap_533 * engaged_gap.gap_533) +
    (engaging_gap.gap_534 * engaged_gap.gap_534) +
    (engaging_gap.gap_535 * engaged_gap.gap_535) +
    (engaging_gap.gap_536 * engaged_gap.gap_536) +
    (engaging_gap.gap_537 * engaged_gap.gap_537) +
    (engaging_gap.gap_538 * engaged_gap.gap_538) +
    (engaging_gap.gap_539 * engaged_gap.gap_539) +
    (engaging_gap.gap_540 * engaged_gap.gap_540) +
    (engaging_gap.gap_541 * engaged_gap.gap_541) +
    (engaging_gap.gap_542 * engaged_gap.gap_542) +
    (engaging_gap.gap_543 * engaged_gap.gap_543) +
    (engaging_gap.gap_544 * engaged_gap.gap_544) +
    (engaging_gap.gap_545 * engaged_gap.gap_545) +
    (engaging_gap.gap_546 * engaged_gap.gap_546) +
    (engaging_gap.gap_547 * engaged_gap.gap_547) +
    (engaging_gap.gap_548 * engaged_gap.gap_548) +
    (engaging_gap.gap_549 * engaged_gap.gap_549) +
    (engaging_gap.gap_550 * engaged_gap.gap_550) +
    (engaging_gap.gap_551 * engaged_gap.gap_551) +
    (engaging_gap.gap_552 * engaged_gap.gap_552) +
    (engaging_gap.gap_553 * engaged_gap.gap_553) +
    (engaging_gap.gap_554 * engaged_gap.gap_554) +
    (engaging_gap.gap_555 * engaged_gap.gap_555) +
    (engaging_gap.gap_556 * engaged_gap.gap_556) +
    (engaging_gap.gap_557 * engaged_gap.gap_557) +
    (engaging_gap.gap_558 * engaged_gap.gap_558) +
    (engaging_gap.gap_559 * engaged_gap.gap_559) +
    (engaging_gap.gap_560 * engaged_gap.gap_560) +
    (engaging_gap.gap_561 * engaged_gap.gap_561) +
    (engaging_gap.gap_562 * engaged_gap.gap_562) +
    (engaging_gap.gap_563 * engaged_gap.gap_563) +
    (engaging_gap.gap_564 * engaged_gap.gap_564) +
    (engaging_gap.gap_565 * engaged_gap.gap_565) +
    (engaging_gap.gap_566 * engaged_gap.gap_566) +
    (engaging_gap.gap_567 * engaged_gap.gap_567) +
    (engaging_gap.gap_568 * engaged_gap.gap_568) +
    (engaging_gap.gap_569 * engaged_gap.gap_569) +
    (engaging_gap.gap_570 * engaged_gap.gap_570) +
    (engaging_gap.gap_571 * engaged_gap.gap_571) +
    (engaging_gap.gap_572 * engaged_gap.gap_572) +
    (engaging_gap.gap_573 * engaged_gap.gap_573) +
    (engaging_gap.gap_574 * engaged_gap.gap_574) +
    (engaging_gap.gap_575 * engaged_gap.gap_575) +
    (engaging_gap.gap_576 * engaged_gap.gap_576) +
    (engaging_gap.gap_577 * engaged_gap.gap_577) +
    (engaging_gap.gap_578 * engaged_gap.gap_578) +
    (engaging_gap.gap_579 * engaged_gap.gap_579) +
    (engaging_gap.gap_580 * engaged_gap.gap_580) +
    (engaging_gap.gap_581 * engaged_gap.gap_581) +
    (engaging_gap.gap_582 * engaged_gap.gap_582) +
    (engaging_gap.gap_583 * engaged_gap.gap_583) +
    (engaging_gap.gap_584 * engaged_gap.gap_584) +
    (engaging_gap.gap_585 * engaged_gap.gap_585) +
    (engaging_gap.gap_586 * engaged_gap.gap_586) +
    (engaging_gap.gap_587 * engaged_gap.gap_587) +
    (engaging_gap.gap_588 * engaged_gap.gap_588) +
    (engaging_gap.gap_589 * engaged_gap.gap_589) +
    (engaging_gap.gap_590 * engaged_gap.gap_590) +
    (engaging_gap.gap_591 * engaged_gap.gap_591) +
    (engaging_gap.gap_592 * engaged_gap.gap_592) +
    (engaging_gap.gap_593 * engaged_gap.gap_593) +
    (engaging_gap.gap_594 * engaged_gap.gap_594) +
    (engaging_gap.gap_595 * engaged_gap.gap_595) +
    (engaging_gap.gap_596 * engaged_gap.gap_596) +
    (engaging_gap.gap_597 * engaged_gap.gap_597) +
    (engaging_gap.gap_598 * engaged_gap.gap_598) +
    (engaging_gap.gap_599 * engaged_gap.gap_599) +
    (engaging_gap.gap_600 * engaged_gap.gap_600) +
    (engaging_gap.gap_601 * engaged_gap.gap_601) +
    (engaging_gap.gap_602 * engaged_gap.gap_602) +
    (engaging_gap.gap_603 * engaged_gap.gap_603) +
    (engaging_gap.gap_604 * engaged_gap.gap_604) +
    (engaging_gap.gap_605 * engaged_gap.gap_605) +
    (engaging_gap.gap_606 * engaged_gap.gap_606) +
    (engaging_gap.gap_607 * engaged_gap.gap_607) +
    (engaging_gap.gap_608 * engaged_gap.gap_608) +
    (engaging_gap.gap_609 * engaged_gap.gap_609) +
    (engaging_gap.gap_610 * engaged_gap.gap_610) +
    (engaging_gap.gap_611 * engaged_gap.gap_611) +
    (engaging_gap.gap_612 * engaged_gap.gap_612) +
    (engaging_gap.gap_613 * engaged_gap.gap_613) +
    (engaging_gap.gap_614 * engaged_gap.gap_614) +
    (engaging_gap.gap_615 * engaged_gap.gap_615) +
    (engaging_gap.gap_616 * engaged_gap.gap_616) +
    (engaging_gap.gap_617 * engaged_gap.gap_617) +
    (engaging_gap.gap_618 * engaged_gap.gap_618) +
    (engaging_gap.gap_619 * engaged_gap.gap_619) +
    (engaging_gap.gap_620 * engaged_gap.gap_620) +
    (engaging_gap.gap_621 * engaged_gap.gap_621) +
    (engaging_gap.gap_622 * engaged_gap.gap_622) +
    (engaging_gap.gap_623 * engaged_gap.gap_623) +
    (engaging_gap.gap_624 * engaged_gap.gap_624) +
    (engaging_gap.gap_625 * engaged_gap.gap_625) +
    (engaging_gap.gap_626 * engaged_gap.gap_626) +
    (engaging_gap.gap_627 * engaged_gap.gap_627) +
    (engaging_gap.gap_628 * engaged_gap.gap_628) +
    (engaging_gap.gap_629 * engaged_gap.gap_629) +
    (engaging_gap.gap_630 * engaged_gap.gap_630) +
    (engaging_gap.gap_631 * engaged_gap.gap_631) +
    (engaging_gap.gap_632 * engaged_gap.gap_632) +
    (engaging_gap.gap_633 * engaged_gap.gap_633) +
    (engaging_gap.gap_634 * engaged_gap.gap_634) +
    (engaging_gap.gap_635 * engaged_gap.gap_635) +
    (engaging_gap.gap_636 * engaged_gap.gap_636) +
    (engaging_gap.gap_637 * engaged_gap.gap_637) +
    (engaging_gap.gap_638 * engaged_gap.gap_638) +
    (engaging_gap.gap_639 * engaged_gap.gap_639) +
    (engaging_gap.gap_640 * engaged_gap.gap_640) +
    (engaging_gap.gap_641 * engaged_gap.gap_641) +
    (engaging_gap.gap_642 * engaged_gap.gap_642) +
    (engaging_gap.gap_643 * engaged_gap.gap_643) +
    (engaging_gap.gap_644 * engaged_gap.gap_644) +
    (engaging_gap.gap_645 * engaged_gap.gap_645) +
    (engaging_gap.gap_646 * engaged_gap.gap_646) +
    (engaging_gap.gap_647 * engaged_gap.gap_647) +
    (engaging_gap.gap_648 * engaged_gap.gap_648) +
    (engaging_gap.gap_649 * engaged_gap.gap_649) +
    (engaging_gap.gap_650 * engaged_gap.gap_650) +
    (engaging_gap.gap_651 * engaged_gap.gap_651) +
    (engaging_gap.gap_652 * engaged_gap.gap_652) +
    (engaging_gap.gap_653 * engaged_gap.gap_653) +
    (engaging_gap.gap_654 * engaged_gap.gap_654) +
    (engaging_gap.gap_655 * engaged_gap.gap_655) +
    (engaging_gap.gap_656 * engaged_gap.gap_656) +
    (engaging_gap.gap_657 * engaged_gap.gap_657) +
    (engaging_gap.gap_658 * engaged_gap.gap_658) +
    (engaging_gap.gap_659 * engaged_gap.gap_659) +
    (engaging_gap.gap_660 * engaged_gap.gap_660) +
    (engaging_gap.gap_661 * engaged_gap.gap_661) +
    (engaging_gap.gap_662 * engaged_gap.gap_662) +
    (engaging_gap.gap_663 * engaged_gap.gap_663) +
    (engaging_gap.gap_664 * engaged_gap.gap_664) +
    (engaging_gap.gap_665 * engaged_gap.gap_665) +
    (engaging_gap.gap_666 * engaged_gap.gap_666) +
    (engaging_gap.gap_667 * engaged_gap.gap_667) +
    (engaging_gap.gap_668 * engaged_gap.gap_668) +
    (engaging_gap.gap_669 * engaged_gap.gap_669) +
    (engaging_gap.gap_670 * engaged_gap.gap_670) +
    (engaging_gap.gap_671 * engaged_gap.gap_671) +
    (engaging_gap.gap_672 * engaged_gap.gap_672) +
    (engaging_gap.gap_673 * engaged_gap.gap_673) +
    (engaging_gap.gap_674 * engaged_gap.gap_674) +
    (engaging_gap.gap_675 * engaged_gap.gap_675) +
    (engaging_gap.gap_676 * engaged_gap.gap_676) +
    (engaging_gap.gap_677 * engaged_gap.gap_677) +
    (engaging_gap.gap_678 * engaged_gap.gap_678) +
    (engaging_gap.gap_679 * engaged_gap.gap_679) +
    (engaging_gap.gap_680 * engaged_gap.gap_680) +
    (engaging_gap.gap_681 * engaged_gap.gap_681) +
    (engaging_gap.gap_682 * engaged_gap.gap_682) +
    (engaging_gap.gap_683 * engaged_gap.gap_683) +
    (engaging_gap.gap_684 * engaged_gap.gap_684) +
    (engaging_gap.gap_685 * engaged_gap.gap_685) +
    (engaging_gap.gap_686 * engaged_gap.gap_686) +
    (engaging_gap.gap_687 * engaged_gap.gap_687) +
    (engaging_gap.gap_688 * engaged_gap.gap_688) +
    (engaging_gap.gap_689 * engaged_gap.gap_689) +
    (engaging_gap.gap_690 * engaged_gap.gap_690) +
    (engaging_gap.gap_691 * engaged_gap.gap_691) +
    (engaging_gap.gap_692 * engaged_gap.gap_692) +
    (engaging_gap.gap_693 * engaged_gap.gap_693) +
    (engaging_gap.gap_694 * engaged_gap.gap_694) +
    (engaging_gap.gap_695 * engaged_gap.gap_695) +
    (engaging_gap.gap_696 * engaged_gap.gap_696) +
    (engaging_gap.gap_697 * engaged_gap.gap_697) +
    (engaging_gap.gap_698 * engaged_gap.gap_698) +
    (engaging_gap.gap_699 * engaged_gap.gap_699) +
    (engaging_gap.gap_700 * engaged_gap.gap_700) +
    (engaging_gap.gap_701 * engaged_gap.gap_701) +
    (engaging_gap.gap_702 * engaged_gap.gap_702) +
    (engaging_gap.gap_703 * engaged_gap.gap_703) +
    (engaging_gap.gap_704 * engaged_gap.gap_704) +
    (engaging_gap.gap_705 * engaged_gap.gap_705) +
    (engaging_gap.gap_706 * engaged_gap.gap_706) +
    (engaging_gap.gap_707 * engaged_gap.gap_707) +
    (engaging_gap.gap_708 * engaged_gap.gap_708) +
    (engaging_gap.gap_709 * engaged_gap.gap_709) +
    (engaging_gap.gap_710 * engaged_gap.gap_710) +
    (engaging_gap.gap_711 * engaged_gap.gap_711) +
    (engaging_gap.gap_712 * engaged_gap.gap_712) +
    (engaging_gap.gap_713 * engaged_gap.gap_713) +
    (engaging_gap.gap_714 * engaged_gap.gap_714) +
    (engaging_gap.gap_715 * engaged_gap.gap_715) +
    (engaging_gap.gap_716 * engaged_gap.gap_716) +
    (engaging_gap.gap_717 * engaged_gap.gap_717) +
    (engaging_gap.gap_718 * engaged_gap.gap_718) +
    (engaging_gap.gap_719 * engaged_gap.gap_719) +
    (engaging_gap.gap_720 * engaged_gap.gap_720) +
    (engaging_gap.gap_721 * engaged_gap.gap_721) +
    (engaging_gap.gap_722 * engaged_gap.gap_722) +
    (engaging_gap.gap_723 * engaged_gap.gap_723) +
    (engaging_gap.gap_724 * engaged_gap.gap_724) +
    (engaging_gap.gap_725 * engaged_gap.gap_725) +
    (engaging_gap.gap_726 * engaged_gap.gap_726) +
    (engaging_gap.gap_727 * engaged_gap.gap_727) +
    (engaging_gap.gap_728 * engaged_gap.gap_728) +
    (engaging_gap.gap_729 * engaged_gap.gap_729) +
    (engaging_gap.gap_730 * engaged_gap.gap_730) +
    (engaging_gap.gap_731 * engaged_gap.gap_731) +
    (engaging_gap.gap_732 * engaged_gap.gap_732) +
    (engaging_gap.gap_733 * engaged_gap.gap_733) +
    (engaging_gap.gap_734 * engaged_gap.gap_734) +
    (engaging_gap.gap_735 * engaged_gap.gap_735) +
    (engaging_gap.gap_736 * engaged_gap.gap_736) +
    (engaging_gap.gap_737 * engaged_gap.gap_737) +
    (engaging_gap.gap_738 * engaged_gap.gap_738) +
    (engaging_gap.gap_739 * engaged_gap.gap_739) +
    (engaging_gap.gap_740 * engaged_gap.gap_740) +
    (engaging_gap.gap_741 * engaged_gap.gap_741) +
    (engaging_gap.gap_742 * engaged_gap.gap_742) +
    (engaging_gap.gap_743 * engaged_gap.gap_743) +
    (engaging_gap.gap_744 * engaged_gap.gap_744) +
    (engaging_gap.gap_745 * engaged_gap.gap_745) +
    (engaging_gap.gap_746 * engaged_gap.gap_746) +
    (engaging_gap.gap_747 * engaged_gap.gap_747) +
    (engaging_gap.gap_748 * engaged_gap.gap_748) +
    (engaging_gap.gap_749 * engaged_gap.gap_749) +
    (engaging_gap.gap_750 * engaged_gap.gap_750) +
    (engaging_gap.gap_751 * engaged_gap.gap_751) +
    (engaging_gap.gap_752 * engaged_gap.gap_752) +
    (engaging_gap.gap_753 * engaged_gap.gap_753) +
    (engaging_gap.gap_754 * engaged_gap.gap_754) +
    (engaging_gap.gap_755 * engaged_gap.gap_755) +
    (engaging_gap.gap_756 * engaged_gap.gap_756) +
    (engaging_gap.gap_757 * engaged_gap.gap_757) +
    (engaging_gap.gap_758 * engaged_gap.gap_758) +
    (engaging_gap.gap_759 * engaged_gap.gap_759) +
    (engaging_gap.gap_760 * engaged_gap.gap_760) +
    (engaging_gap.gap_761 * engaged_gap.gap_761) +
    (engaging_gap.gap_762 * engaged_gap.gap_762) +
    (engaging_gap.gap_763 * engaged_gap.gap_763) +
    (engaging_gap.gap_764 * engaged_gap.gap_764) +
    (engaging_gap.gap_765 * engaged_gap.gap_765) +
    (engaging_gap.gap_766 * engaged_gap.gap_766) +
    (engaging_gap.gap_767 * engaged_gap.gap_767)
  ) as dot_product_of_engaging_user_and_engaged_user
from {table_name} t
left join user_tweet_vectors engaging_gap on t.engaging_user_id = engaging_gap.user_id
left join user_tweet_vectors engaged_gap on t.engaged_user_id = engaged_gap.user_id
order by t.tweet_id, t.engaging_user_id
"""


if __name__ == "__main__":
    BertSimilarityBetweenEngagingAndEngagedTweetsVectorsFeature.main()
