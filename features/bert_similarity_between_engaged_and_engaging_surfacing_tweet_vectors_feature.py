from typing import List, Tuple

from google.cloud import bigquery, bigquery_storage_v1beta1
import pandas as pd

from base import BaseFeature, TESTING, PROJECT_ID, reduce_mem_usage


class BertSimilarityBetweenEngagedAndEngagingSurfacingTweetVectorsFeature(BaseFeature):
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

        bqclient = bigquery.Client(project=PROJECT_ID)
        bqstorageclient = bigquery_storage_v1beta1.BigQueryStorageClient()
        df = (
            bqclient.query(query)
            .result()
            .to_dataframe(bqstorage_client=bqstorageclient)
        )
        return df

_QUERY = r"""
with surfacing_tweets as (
  select tweet_id, engaging_user_id
  from `recsys2020.training` t
  group by tweet_id, engaging_user_id
),
user_surfacing_tweet_vectors as (
  select
    engaging_user_id as user_id,
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
  from surfacing_tweets
  inner join `recsys2020.pretrained_bert_gap` gap on surfacing_tweets.tweet_id = gap.tweet_id
  group by user_id
)

select
  t.tweet_id,
  t.engaging_user_id,
  1.0 / 768 * (
    (engaged_user_surfacing_tweet_vectors.gap_0 * user_surfacing_tweet_vectors.gap_0) +
    (engaged_user_surfacing_tweet_vectors.gap_1 * user_surfacing_tweet_vectors.gap_1) +
    (engaged_user_surfacing_tweet_vectors.gap_2 * user_surfacing_tweet_vectors.gap_2) +
    (engaged_user_surfacing_tweet_vectors.gap_3 * user_surfacing_tweet_vectors.gap_3) +
    (engaged_user_surfacing_tweet_vectors.gap_4 * user_surfacing_tweet_vectors.gap_4) +
    (engaged_user_surfacing_tweet_vectors.gap_5 * user_surfacing_tweet_vectors.gap_5) +
    (engaged_user_surfacing_tweet_vectors.gap_6 * user_surfacing_tweet_vectors.gap_6) +
    (engaged_user_surfacing_tweet_vectors.gap_7 * user_surfacing_tweet_vectors.gap_7) +
    (engaged_user_surfacing_tweet_vectors.gap_8 * user_surfacing_tweet_vectors.gap_8) +
    (engaged_user_surfacing_tweet_vectors.gap_9 * user_surfacing_tweet_vectors.gap_9) +
    (engaged_user_surfacing_tweet_vectors.gap_10 * user_surfacing_tweet_vectors.gap_10) +
    (engaged_user_surfacing_tweet_vectors.gap_11 * user_surfacing_tweet_vectors.gap_11) +
    (engaged_user_surfacing_tweet_vectors.gap_12 * user_surfacing_tweet_vectors.gap_12) +
    (engaged_user_surfacing_tweet_vectors.gap_13 * user_surfacing_tweet_vectors.gap_13) +
    (engaged_user_surfacing_tweet_vectors.gap_14 * user_surfacing_tweet_vectors.gap_14) +
    (engaged_user_surfacing_tweet_vectors.gap_15 * user_surfacing_tweet_vectors.gap_15) +
    (engaged_user_surfacing_tweet_vectors.gap_16 * user_surfacing_tweet_vectors.gap_16) +
    (engaged_user_surfacing_tweet_vectors.gap_17 * user_surfacing_tweet_vectors.gap_17) +
    (engaged_user_surfacing_tweet_vectors.gap_18 * user_surfacing_tweet_vectors.gap_18) +
    (engaged_user_surfacing_tweet_vectors.gap_19 * user_surfacing_tweet_vectors.gap_19) +
    (engaged_user_surfacing_tweet_vectors.gap_20 * user_surfacing_tweet_vectors.gap_20) +
    (engaged_user_surfacing_tweet_vectors.gap_21 * user_surfacing_tweet_vectors.gap_21) +
    (engaged_user_surfacing_tweet_vectors.gap_22 * user_surfacing_tweet_vectors.gap_22) +
    (engaged_user_surfacing_tweet_vectors.gap_23 * user_surfacing_tweet_vectors.gap_23) +
    (engaged_user_surfacing_tweet_vectors.gap_24 * user_surfacing_tweet_vectors.gap_24) +
    (engaged_user_surfacing_tweet_vectors.gap_25 * user_surfacing_tweet_vectors.gap_25) +
    (engaged_user_surfacing_tweet_vectors.gap_26 * user_surfacing_tweet_vectors.gap_26) +
    (engaged_user_surfacing_tweet_vectors.gap_27 * user_surfacing_tweet_vectors.gap_27) +
    (engaged_user_surfacing_tweet_vectors.gap_28 * user_surfacing_tweet_vectors.gap_28) +
    (engaged_user_surfacing_tweet_vectors.gap_29 * user_surfacing_tweet_vectors.gap_29) +
    (engaged_user_surfacing_tweet_vectors.gap_30 * user_surfacing_tweet_vectors.gap_30) +
    (engaged_user_surfacing_tweet_vectors.gap_31 * user_surfacing_tweet_vectors.gap_31) +
    (engaged_user_surfacing_tweet_vectors.gap_32 * user_surfacing_tweet_vectors.gap_32) +
    (engaged_user_surfacing_tweet_vectors.gap_33 * user_surfacing_tweet_vectors.gap_33) +
    (engaged_user_surfacing_tweet_vectors.gap_34 * user_surfacing_tweet_vectors.gap_34) +
    (engaged_user_surfacing_tweet_vectors.gap_35 * user_surfacing_tweet_vectors.gap_35) +
    (engaged_user_surfacing_tweet_vectors.gap_36 * user_surfacing_tweet_vectors.gap_36) +
    (engaged_user_surfacing_tweet_vectors.gap_37 * user_surfacing_tweet_vectors.gap_37) +
    (engaged_user_surfacing_tweet_vectors.gap_38 * user_surfacing_tweet_vectors.gap_38) +
    (engaged_user_surfacing_tweet_vectors.gap_39 * user_surfacing_tweet_vectors.gap_39) +
    (engaged_user_surfacing_tweet_vectors.gap_40 * user_surfacing_tweet_vectors.gap_40) +
    (engaged_user_surfacing_tweet_vectors.gap_41 * user_surfacing_tweet_vectors.gap_41) +
    (engaged_user_surfacing_tweet_vectors.gap_42 * user_surfacing_tweet_vectors.gap_42) +
    (engaged_user_surfacing_tweet_vectors.gap_43 * user_surfacing_tweet_vectors.gap_43) +
    (engaged_user_surfacing_tweet_vectors.gap_44 * user_surfacing_tweet_vectors.gap_44) +
    (engaged_user_surfacing_tweet_vectors.gap_45 * user_surfacing_tweet_vectors.gap_45) +
    (engaged_user_surfacing_tweet_vectors.gap_46 * user_surfacing_tweet_vectors.gap_46) +
    (engaged_user_surfacing_tweet_vectors.gap_47 * user_surfacing_tweet_vectors.gap_47) +
    (engaged_user_surfacing_tweet_vectors.gap_48 * user_surfacing_tweet_vectors.gap_48) +
    (engaged_user_surfacing_tweet_vectors.gap_49 * user_surfacing_tweet_vectors.gap_49) +
    (engaged_user_surfacing_tweet_vectors.gap_50 * user_surfacing_tweet_vectors.gap_50) +
    (engaged_user_surfacing_tweet_vectors.gap_51 * user_surfacing_tweet_vectors.gap_51) +
    (engaged_user_surfacing_tweet_vectors.gap_52 * user_surfacing_tweet_vectors.gap_52) +
    (engaged_user_surfacing_tweet_vectors.gap_53 * user_surfacing_tweet_vectors.gap_53) +
    (engaged_user_surfacing_tweet_vectors.gap_54 * user_surfacing_tweet_vectors.gap_54) +
    (engaged_user_surfacing_tweet_vectors.gap_55 * user_surfacing_tweet_vectors.gap_55) +
    (engaged_user_surfacing_tweet_vectors.gap_56 * user_surfacing_tweet_vectors.gap_56) +
    (engaged_user_surfacing_tweet_vectors.gap_57 * user_surfacing_tweet_vectors.gap_57) +
    (engaged_user_surfacing_tweet_vectors.gap_58 * user_surfacing_tweet_vectors.gap_58) +
    (engaged_user_surfacing_tweet_vectors.gap_59 * user_surfacing_tweet_vectors.gap_59) +
    (engaged_user_surfacing_tweet_vectors.gap_60 * user_surfacing_tweet_vectors.gap_60) +
    (engaged_user_surfacing_tweet_vectors.gap_61 * user_surfacing_tweet_vectors.gap_61) +
    (engaged_user_surfacing_tweet_vectors.gap_62 * user_surfacing_tweet_vectors.gap_62) +
    (engaged_user_surfacing_tweet_vectors.gap_63 * user_surfacing_tweet_vectors.gap_63) +
    (engaged_user_surfacing_tweet_vectors.gap_64 * user_surfacing_tweet_vectors.gap_64) +
    (engaged_user_surfacing_tweet_vectors.gap_65 * user_surfacing_tweet_vectors.gap_65) +
    (engaged_user_surfacing_tweet_vectors.gap_66 * user_surfacing_tweet_vectors.gap_66) +
    (engaged_user_surfacing_tweet_vectors.gap_67 * user_surfacing_tweet_vectors.gap_67) +
    (engaged_user_surfacing_tweet_vectors.gap_68 * user_surfacing_tweet_vectors.gap_68) +
    (engaged_user_surfacing_tweet_vectors.gap_69 * user_surfacing_tweet_vectors.gap_69) +
    (engaged_user_surfacing_tweet_vectors.gap_70 * user_surfacing_tweet_vectors.gap_70) +
    (engaged_user_surfacing_tweet_vectors.gap_71 * user_surfacing_tweet_vectors.gap_71) +
    (engaged_user_surfacing_tweet_vectors.gap_72 * user_surfacing_tweet_vectors.gap_72) +
    (engaged_user_surfacing_tweet_vectors.gap_73 * user_surfacing_tweet_vectors.gap_73) +
    (engaged_user_surfacing_tweet_vectors.gap_74 * user_surfacing_tweet_vectors.gap_74) +
    (engaged_user_surfacing_tweet_vectors.gap_75 * user_surfacing_tweet_vectors.gap_75) +
    (engaged_user_surfacing_tweet_vectors.gap_76 * user_surfacing_tweet_vectors.gap_76) +
    (engaged_user_surfacing_tweet_vectors.gap_77 * user_surfacing_tweet_vectors.gap_77) +
    (engaged_user_surfacing_tweet_vectors.gap_78 * user_surfacing_tweet_vectors.gap_78) +
    (engaged_user_surfacing_tweet_vectors.gap_79 * user_surfacing_tweet_vectors.gap_79) +
    (engaged_user_surfacing_tweet_vectors.gap_80 * user_surfacing_tweet_vectors.gap_80) +
    (engaged_user_surfacing_tweet_vectors.gap_81 * user_surfacing_tweet_vectors.gap_81) +
    (engaged_user_surfacing_tweet_vectors.gap_82 * user_surfacing_tweet_vectors.gap_82) +
    (engaged_user_surfacing_tweet_vectors.gap_83 * user_surfacing_tweet_vectors.gap_83) +
    (engaged_user_surfacing_tweet_vectors.gap_84 * user_surfacing_tweet_vectors.gap_84) +
    (engaged_user_surfacing_tweet_vectors.gap_85 * user_surfacing_tweet_vectors.gap_85) +
    (engaged_user_surfacing_tweet_vectors.gap_86 * user_surfacing_tweet_vectors.gap_86) +
    (engaged_user_surfacing_tweet_vectors.gap_87 * user_surfacing_tweet_vectors.gap_87) +
    (engaged_user_surfacing_tweet_vectors.gap_88 * user_surfacing_tweet_vectors.gap_88) +
    (engaged_user_surfacing_tweet_vectors.gap_89 * user_surfacing_tweet_vectors.gap_89) +
    (engaged_user_surfacing_tweet_vectors.gap_90 * user_surfacing_tweet_vectors.gap_90) +
    (engaged_user_surfacing_tweet_vectors.gap_91 * user_surfacing_tweet_vectors.gap_91) +
    (engaged_user_surfacing_tweet_vectors.gap_92 * user_surfacing_tweet_vectors.gap_92) +
    (engaged_user_surfacing_tweet_vectors.gap_93 * user_surfacing_tweet_vectors.gap_93) +
    (engaged_user_surfacing_tweet_vectors.gap_94 * user_surfacing_tweet_vectors.gap_94) +
    (engaged_user_surfacing_tweet_vectors.gap_95 * user_surfacing_tweet_vectors.gap_95) +
    (engaged_user_surfacing_tweet_vectors.gap_96 * user_surfacing_tweet_vectors.gap_96) +
    (engaged_user_surfacing_tweet_vectors.gap_97 * user_surfacing_tweet_vectors.gap_97) +
    (engaged_user_surfacing_tweet_vectors.gap_98 * user_surfacing_tweet_vectors.gap_98) +
    (engaged_user_surfacing_tweet_vectors.gap_99 * user_surfacing_tweet_vectors.gap_99) +
    (engaged_user_surfacing_tweet_vectors.gap_100 * user_surfacing_tweet_vectors.gap_100) +
    (engaged_user_surfacing_tweet_vectors.gap_101 * user_surfacing_tweet_vectors.gap_101) +
    (engaged_user_surfacing_tweet_vectors.gap_102 * user_surfacing_tweet_vectors.gap_102) +
    (engaged_user_surfacing_tweet_vectors.gap_103 * user_surfacing_tweet_vectors.gap_103) +
    (engaged_user_surfacing_tweet_vectors.gap_104 * user_surfacing_tweet_vectors.gap_104) +
    (engaged_user_surfacing_tweet_vectors.gap_105 * user_surfacing_tweet_vectors.gap_105) +
    (engaged_user_surfacing_tweet_vectors.gap_106 * user_surfacing_tweet_vectors.gap_106) +
    (engaged_user_surfacing_tweet_vectors.gap_107 * user_surfacing_tweet_vectors.gap_107) +
    (engaged_user_surfacing_tweet_vectors.gap_108 * user_surfacing_tweet_vectors.gap_108) +
    (engaged_user_surfacing_tweet_vectors.gap_109 * user_surfacing_tweet_vectors.gap_109) +
    (engaged_user_surfacing_tweet_vectors.gap_110 * user_surfacing_tweet_vectors.gap_110) +
    (engaged_user_surfacing_tweet_vectors.gap_111 * user_surfacing_tweet_vectors.gap_111) +
    (engaged_user_surfacing_tweet_vectors.gap_112 * user_surfacing_tweet_vectors.gap_112) +
    (engaged_user_surfacing_tweet_vectors.gap_113 * user_surfacing_tweet_vectors.gap_113) +
    (engaged_user_surfacing_tweet_vectors.gap_114 * user_surfacing_tweet_vectors.gap_114) +
    (engaged_user_surfacing_tweet_vectors.gap_115 * user_surfacing_tweet_vectors.gap_115) +
    (engaged_user_surfacing_tweet_vectors.gap_116 * user_surfacing_tweet_vectors.gap_116) +
    (engaged_user_surfacing_tweet_vectors.gap_117 * user_surfacing_tweet_vectors.gap_117) +
    (engaged_user_surfacing_tweet_vectors.gap_118 * user_surfacing_tweet_vectors.gap_118) +
    (engaged_user_surfacing_tweet_vectors.gap_119 * user_surfacing_tweet_vectors.gap_119) +
    (engaged_user_surfacing_tweet_vectors.gap_120 * user_surfacing_tweet_vectors.gap_120) +
    (engaged_user_surfacing_tweet_vectors.gap_121 * user_surfacing_tweet_vectors.gap_121) +
    (engaged_user_surfacing_tweet_vectors.gap_122 * user_surfacing_tweet_vectors.gap_122) +
    (engaged_user_surfacing_tweet_vectors.gap_123 * user_surfacing_tweet_vectors.gap_123) +
    (engaged_user_surfacing_tweet_vectors.gap_124 * user_surfacing_tweet_vectors.gap_124) +
    (engaged_user_surfacing_tweet_vectors.gap_125 * user_surfacing_tweet_vectors.gap_125) +
    (engaged_user_surfacing_tweet_vectors.gap_126 * user_surfacing_tweet_vectors.gap_126) +
    (engaged_user_surfacing_tweet_vectors.gap_127 * user_surfacing_tweet_vectors.gap_127) +
    (engaged_user_surfacing_tweet_vectors.gap_128 * user_surfacing_tweet_vectors.gap_128) +
    (engaged_user_surfacing_tweet_vectors.gap_129 * user_surfacing_tweet_vectors.gap_129) +
    (engaged_user_surfacing_tweet_vectors.gap_130 * user_surfacing_tweet_vectors.gap_130) +
    (engaged_user_surfacing_tweet_vectors.gap_131 * user_surfacing_tweet_vectors.gap_131) +
    (engaged_user_surfacing_tweet_vectors.gap_132 * user_surfacing_tweet_vectors.gap_132) +
    (engaged_user_surfacing_tweet_vectors.gap_133 * user_surfacing_tweet_vectors.gap_133) +
    (engaged_user_surfacing_tweet_vectors.gap_134 * user_surfacing_tweet_vectors.gap_134) +
    (engaged_user_surfacing_tweet_vectors.gap_135 * user_surfacing_tweet_vectors.gap_135) +
    (engaged_user_surfacing_tweet_vectors.gap_136 * user_surfacing_tweet_vectors.gap_136) +
    (engaged_user_surfacing_tweet_vectors.gap_137 * user_surfacing_tweet_vectors.gap_137) +
    (engaged_user_surfacing_tweet_vectors.gap_138 * user_surfacing_tweet_vectors.gap_138) +
    (engaged_user_surfacing_tweet_vectors.gap_139 * user_surfacing_tweet_vectors.gap_139) +
    (engaged_user_surfacing_tweet_vectors.gap_140 * user_surfacing_tweet_vectors.gap_140) +
    (engaged_user_surfacing_tweet_vectors.gap_141 * user_surfacing_tweet_vectors.gap_141) +
    (engaged_user_surfacing_tweet_vectors.gap_142 * user_surfacing_tweet_vectors.gap_142) +
    (engaged_user_surfacing_tweet_vectors.gap_143 * user_surfacing_tweet_vectors.gap_143) +
    (engaged_user_surfacing_tweet_vectors.gap_144 * user_surfacing_tweet_vectors.gap_144) +
    (engaged_user_surfacing_tweet_vectors.gap_145 * user_surfacing_tweet_vectors.gap_145) +
    (engaged_user_surfacing_tweet_vectors.gap_146 * user_surfacing_tweet_vectors.gap_146) +
    (engaged_user_surfacing_tweet_vectors.gap_147 * user_surfacing_tweet_vectors.gap_147) +
    (engaged_user_surfacing_tweet_vectors.gap_148 * user_surfacing_tweet_vectors.gap_148) +
    (engaged_user_surfacing_tweet_vectors.gap_149 * user_surfacing_tweet_vectors.gap_149) +
    (engaged_user_surfacing_tweet_vectors.gap_150 * user_surfacing_tweet_vectors.gap_150) +
    (engaged_user_surfacing_tweet_vectors.gap_151 * user_surfacing_tweet_vectors.gap_151) +
    (engaged_user_surfacing_tweet_vectors.gap_152 * user_surfacing_tweet_vectors.gap_152) +
    (engaged_user_surfacing_tweet_vectors.gap_153 * user_surfacing_tweet_vectors.gap_153) +
    (engaged_user_surfacing_tweet_vectors.gap_154 * user_surfacing_tweet_vectors.gap_154) +
    (engaged_user_surfacing_tweet_vectors.gap_155 * user_surfacing_tweet_vectors.gap_155) +
    (engaged_user_surfacing_tweet_vectors.gap_156 * user_surfacing_tweet_vectors.gap_156) +
    (engaged_user_surfacing_tweet_vectors.gap_157 * user_surfacing_tweet_vectors.gap_157) +
    (engaged_user_surfacing_tweet_vectors.gap_158 * user_surfacing_tweet_vectors.gap_158) +
    (engaged_user_surfacing_tweet_vectors.gap_159 * user_surfacing_tweet_vectors.gap_159) +
    (engaged_user_surfacing_tweet_vectors.gap_160 * user_surfacing_tweet_vectors.gap_160) +
    (engaged_user_surfacing_tweet_vectors.gap_161 * user_surfacing_tweet_vectors.gap_161) +
    (engaged_user_surfacing_tweet_vectors.gap_162 * user_surfacing_tweet_vectors.gap_162) +
    (engaged_user_surfacing_tweet_vectors.gap_163 * user_surfacing_tweet_vectors.gap_163) +
    (engaged_user_surfacing_tweet_vectors.gap_164 * user_surfacing_tweet_vectors.gap_164) +
    (engaged_user_surfacing_tweet_vectors.gap_165 * user_surfacing_tweet_vectors.gap_165) +
    (engaged_user_surfacing_tweet_vectors.gap_166 * user_surfacing_tweet_vectors.gap_166) +
    (engaged_user_surfacing_tweet_vectors.gap_167 * user_surfacing_tweet_vectors.gap_167) +
    (engaged_user_surfacing_tweet_vectors.gap_168 * user_surfacing_tweet_vectors.gap_168) +
    (engaged_user_surfacing_tweet_vectors.gap_169 * user_surfacing_tweet_vectors.gap_169) +
    (engaged_user_surfacing_tweet_vectors.gap_170 * user_surfacing_tweet_vectors.gap_170) +
    (engaged_user_surfacing_tweet_vectors.gap_171 * user_surfacing_tweet_vectors.gap_171) +
    (engaged_user_surfacing_tweet_vectors.gap_172 * user_surfacing_tweet_vectors.gap_172) +
    (engaged_user_surfacing_tweet_vectors.gap_173 * user_surfacing_tweet_vectors.gap_173) +
    (engaged_user_surfacing_tweet_vectors.gap_174 * user_surfacing_tweet_vectors.gap_174) +
    (engaged_user_surfacing_tweet_vectors.gap_175 * user_surfacing_tweet_vectors.gap_175) +
    (engaged_user_surfacing_tweet_vectors.gap_176 * user_surfacing_tweet_vectors.gap_176) +
    (engaged_user_surfacing_tweet_vectors.gap_177 * user_surfacing_tweet_vectors.gap_177) +
    (engaged_user_surfacing_tweet_vectors.gap_178 * user_surfacing_tweet_vectors.gap_178) +
    (engaged_user_surfacing_tweet_vectors.gap_179 * user_surfacing_tweet_vectors.gap_179) +
    (engaged_user_surfacing_tweet_vectors.gap_180 * user_surfacing_tweet_vectors.gap_180) +
    (engaged_user_surfacing_tweet_vectors.gap_181 * user_surfacing_tweet_vectors.gap_181) +
    (engaged_user_surfacing_tweet_vectors.gap_182 * user_surfacing_tweet_vectors.gap_182) +
    (engaged_user_surfacing_tweet_vectors.gap_183 * user_surfacing_tweet_vectors.gap_183) +
    (engaged_user_surfacing_tweet_vectors.gap_184 * user_surfacing_tweet_vectors.gap_184) +
    (engaged_user_surfacing_tweet_vectors.gap_185 * user_surfacing_tweet_vectors.gap_185) +
    (engaged_user_surfacing_tweet_vectors.gap_186 * user_surfacing_tweet_vectors.gap_186) +
    (engaged_user_surfacing_tweet_vectors.gap_187 * user_surfacing_tweet_vectors.gap_187) +
    (engaged_user_surfacing_tweet_vectors.gap_188 * user_surfacing_tweet_vectors.gap_188) +
    (engaged_user_surfacing_tweet_vectors.gap_189 * user_surfacing_tweet_vectors.gap_189) +
    (engaged_user_surfacing_tweet_vectors.gap_190 * user_surfacing_tweet_vectors.gap_190) +
    (engaged_user_surfacing_tweet_vectors.gap_191 * user_surfacing_tweet_vectors.gap_191) +
    (engaged_user_surfacing_tweet_vectors.gap_192 * user_surfacing_tweet_vectors.gap_192) +
    (engaged_user_surfacing_tweet_vectors.gap_193 * user_surfacing_tweet_vectors.gap_193) +
    (engaged_user_surfacing_tweet_vectors.gap_194 * user_surfacing_tweet_vectors.gap_194) +
    (engaged_user_surfacing_tweet_vectors.gap_195 * user_surfacing_tweet_vectors.gap_195) +
    (engaged_user_surfacing_tweet_vectors.gap_196 * user_surfacing_tweet_vectors.gap_196) +
    (engaged_user_surfacing_tweet_vectors.gap_197 * user_surfacing_tweet_vectors.gap_197) +
    (engaged_user_surfacing_tweet_vectors.gap_198 * user_surfacing_tweet_vectors.gap_198) +
    (engaged_user_surfacing_tweet_vectors.gap_199 * user_surfacing_tweet_vectors.gap_199) +
    (engaged_user_surfacing_tweet_vectors.gap_200 * user_surfacing_tweet_vectors.gap_200) +
    (engaged_user_surfacing_tweet_vectors.gap_201 * user_surfacing_tweet_vectors.gap_201) +
    (engaged_user_surfacing_tweet_vectors.gap_202 * user_surfacing_tweet_vectors.gap_202) +
    (engaged_user_surfacing_tweet_vectors.gap_203 * user_surfacing_tweet_vectors.gap_203) +
    (engaged_user_surfacing_tweet_vectors.gap_204 * user_surfacing_tweet_vectors.gap_204) +
    (engaged_user_surfacing_tweet_vectors.gap_205 * user_surfacing_tweet_vectors.gap_205) +
    (engaged_user_surfacing_tweet_vectors.gap_206 * user_surfacing_tweet_vectors.gap_206) +
    (engaged_user_surfacing_tweet_vectors.gap_207 * user_surfacing_tweet_vectors.gap_207) +
    (engaged_user_surfacing_tweet_vectors.gap_208 * user_surfacing_tweet_vectors.gap_208) +
    (engaged_user_surfacing_tweet_vectors.gap_209 * user_surfacing_tweet_vectors.gap_209) +
    (engaged_user_surfacing_tweet_vectors.gap_210 * user_surfacing_tweet_vectors.gap_210) +
    (engaged_user_surfacing_tweet_vectors.gap_211 * user_surfacing_tweet_vectors.gap_211) +
    (engaged_user_surfacing_tweet_vectors.gap_212 * user_surfacing_tweet_vectors.gap_212) +
    (engaged_user_surfacing_tweet_vectors.gap_213 * user_surfacing_tweet_vectors.gap_213) +
    (engaged_user_surfacing_tweet_vectors.gap_214 * user_surfacing_tweet_vectors.gap_214) +
    (engaged_user_surfacing_tweet_vectors.gap_215 * user_surfacing_tweet_vectors.gap_215) +
    (engaged_user_surfacing_tweet_vectors.gap_216 * user_surfacing_tweet_vectors.gap_216) +
    (engaged_user_surfacing_tweet_vectors.gap_217 * user_surfacing_tweet_vectors.gap_217) +
    (engaged_user_surfacing_tweet_vectors.gap_218 * user_surfacing_tweet_vectors.gap_218) +
    (engaged_user_surfacing_tweet_vectors.gap_219 * user_surfacing_tweet_vectors.gap_219) +
    (engaged_user_surfacing_tweet_vectors.gap_220 * user_surfacing_tweet_vectors.gap_220) +
    (engaged_user_surfacing_tweet_vectors.gap_221 * user_surfacing_tweet_vectors.gap_221) +
    (engaged_user_surfacing_tweet_vectors.gap_222 * user_surfacing_tweet_vectors.gap_222) +
    (engaged_user_surfacing_tweet_vectors.gap_223 * user_surfacing_tweet_vectors.gap_223) +
    (engaged_user_surfacing_tweet_vectors.gap_224 * user_surfacing_tweet_vectors.gap_224) +
    (engaged_user_surfacing_tweet_vectors.gap_225 * user_surfacing_tweet_vectors.gap_225) +
    (engaged_user_surfacing_tweet_vectors.gap_226 * user_surfacing_tweet_vectors.gap_226) +
    (engaged_user_surfacing_tweet_vectors.gap_227 * user_surfacing_tweet_vectors.gap_227) +
    (engaged_user_surfacing_tweet_vectors.gap_228 * user_surfacing_tweet_vectors.gap_228) +
    (engaged_user_surfacing_tweet_vectors.gap_229 * user_surfacing_tweet_vectors.gap_229) +
    (engaged_user_surfacing_tweet_vectors.gap_230 * user_surfacing_tweet_vectors.gap_230) +
    (engaged_user_surfacing_tweet_vectors.gap_231 * user_surfacing_tweet_vectors.gap_231) +
    (engaged_user_surfacing_tweet_vectors.gap_232 * user_surfacing_tweet_vectors.gap_232) +
    (engaged_user_surfacing_tweet_vectors.gap_233 * user_surfacing_tweet_vectors.gap_233) +
    (engaged_user_surfacing_tweet_vectors.gap_234 * user_surfacing_tweet_vectors.gap_234) +
    (engaged_user_surfacing_tweet_vectors.gap_235 * user_surfacing_tweet_vectors.gap_235) +
    (engaged_user_surfacing_tweet_vectors.gap_236 * user_surfacing_tweet_vectors.gap_236) +
    (engaged_user_surfacing_tweet_vectors.gap_237 * user_surfacing_tweet_vectors.gap_237) +
    (engaged_user_surfacing_tweet_vectors.gap_238 * user_surfacing_tweet_vectors.gap_238) +
    (engaged_user_surfacing_tweet_vectors.gap_239 * user_surfacing_tweet_vectors.gap_239) +
    (engaged_user_surfacing_tweet_vectors.gap_240 * user_surfacing_tweet_vectors.gap_240) +
    (engaged_user_surfacing_tweet_vectors.gap_241 * user_surfacing_tweet_vectors.gap_241) +
    (engaged_user_surfacing_tweet_vectors.gap_242 * user_surfacing_tweet_vectors.gap_242) +
    (engaged_user_surfacing_tweet_vectors.gap_243 * user_surfacing_tweet_vectors.gap_243) +
    (engaged_user_surfacing_tweet_vectors.gap_244 * user_surfacing_tweet_vectors.gap_244) +
    (engaged_user_surfacing_tweet_vectors.gap_245 * user_surfacing_tweet_vectors.gap_245) +
    (engaged_user_surfacing_tweet_vectors.gap_246 * user_surfacing_tweet_vectors.gap_246) +
    (engaged_user_surfacing_tweet_vectors.gap_247 * user_surfacing_tweet_vectors.gap_247) +
    (engaged_user_surfacing_tweet_vectors.gap_248 * user_surfacing_tweet_vectors.gap_248) +
    (engaged_user_surfacing_tweet_vectors.gap_249 * user_surfacing_tweet_vectors.gap_249) +
    (engaged_user_surfacing_tweet_vectors.gap_250 * user_surfacing_tweet_vectors.gap_250) +
    (engaged_user_surfacing_tweet_vectors.gap_251 * user_surfacing_tweet_vectors.gap_251) +
    (engaged_user_surfacing_tweet_vectors.gap_252 * user_surfacing_tweet_vectors.gap_252) +
    (engaged_user_surfacing_tweet_vectors.gap_253 * user_surfacing_tweet_vectors.gap_253) +
    (engaged_user_surfacing_tweet_vectors.gap_254 * user_surfacing_tweet_vectors.gap_254) +
    (engaged_user_surfacing_tweet_vectors.gap_255 * user_surfacing_tweet_vectors.gap_255) +
    (engaged_user_surfacing_tweet_vectors.gap_256 * user_surfacing_tweet_vectors.gap_256) +
    (engaged_user_surfacing_tweet_vectors.gap_257 * user_surfacing_tweet_vectors.gap_257) +
    (engaged_user_surfacing_tweet_vectors.gap_258 * user_surfacing_tweet_vectors.gap_258) +
    (engaged_user_surfacing_tweet_vectors.gap_259 * user_surfacing_tweet_vectors.gap_259) +
    (engaged_user_surfacing_tweet_vectors.gap_260 * user_surfacing_tweet_vectors.gap_260) +
    (engaged_user_surfacing_tweet_vectors.gap_261 * user_surfacing_tweet_vectors.gap_261) +
    (engaged_user_surfacing_tweet_vectors.gap_262 * user_surfacing_tweet_vectors.gap_262) +
    (engaged_user_surfacing_tweet_vectors.gap_263 * user_surfacing_tweet_vectors.gap_263) +
    (engaged_user_surfacing_tweet_vectors.gap_264 * user_surfacing_tweet_vectors.gap_264) +
    (engaged_user_surfacing_tweet_vectors.gap_265 * user_surfacing_tweet_vectors.gap_265) +
    (engaged_user_surfacing_tweet_vectors.gap_266 * user_surfacing_tweet_vectors.gap_266) +
    (engaged_user_surfacing_tweet_vectors.gap_267 * user_surfacing_tweet_vectors.gap_267) +
    (engaged_user_surfacing_tweet_vectors.gap_268 * user_surfacing_tweet_vectors.gap_268) +
    (engaged_user_surfacing_tweet_vectors.gap_269 * user_surfacing_tweet_vectors.gap_269) +
    (engaged_user_surfacing_tweet_vectors.gap_270 * user_surfacing_tweet_vectors.gap_270) +
    (engaged_user_surfacing_tweet_vectors.gap_271 * user_surfacing_tweet_vectors.gap_271) +
    (engaged_user_surfacing_tweet_vectors.gap_272 * user_surfacing_tweet_vectors.gap_272) +
    (engaged_user_surfacing_tweet_vectors.gap_273 * user_surfacing_tweet_vectors.gap_273) +
    (engaged_user_surfacing_tweet_vectors.gap_274 * user_surfacing_tweet_vectors.gap_274) +
    (engaged_user_surfacing_tweet_vectors.gap_275 * user_surfacing_tweet_vectors.gap_275) +
    (engaged_user_surfacing_tweet_vectors.gap_276 * user_surfacing_tweet_vectors.gap_276) +
    (engaged_user_surfacing_tweet_vectors.gap_277 * user_surfacing_tweet_vectors.gap_277) +
    (engaged_user_surfacing_tweet_vectors.gap_278 * user_surfacing_tweet_vectors.gap_278) +
    (engaged_user_surfacing_tweet_vectors.gap_279 * user_surfacing_tweet_vectors.gap_279) +
    (engaged_user_surfacing_tweet_vectors.gap_280 * user_surfacing_tweet_vectors.gap_280) +
    (engaged_user_surfacing_tweet_vectors.gap_281 * user_surfacing_tweet_vectors.gap_281) +
    (engaged_user_surfacing_tweet_vectors.gap_282 * user_surfacing_tweet_vectors.gap_282) +
    (engaged_user_surfacing_tweet_vectors.gap_283 * user_surfacing_tweet_vectors.gap_283) +
    (engaged_user_surfacing_tweet_vectors.gap_284 * user_surfacing_tweet_vectors.gap_284) +
    (engaged_user_surfacing_tweet_vectors.gap_285 * user_surfacing_tweet_vectors.gap_285) +
    (engaged_user_surfacing_tweet_vectors.gap_286 * user_surfacing_tweet_vectors.gap_286) +
    (engaged_user_surfacing_tweet_vectors.gap_287 * user_surfacing_tweet_vectors.gap_287) +
    (engaged_user_surfacing_tweet_vectors.gap_288 * user_surfacing_tweet_vectors.gap_288) +
    (engaged_user_surfacing_tweet_vectors.gap_289 * user_surfacing_tweet_vectors.gap_289) +
    (engaged_user_surfacing_tweet_vectors.gap_290 * user_surfacing_tweet_vectors.gap_290) +
    (engaged_user_surfacing_tweet_vectors.gap_291 * user_surfacing_tweet_vectors.gap_291) +
    (engaged_user_surfacing_tweet_vectors.gap_292 * user_surfacing_tweet_vectors.gap_292) +
    (engaged_user_surfacing_tweet_vectors.gap_293 * user_surfacing_tweet_vectors.gap_293) +
    (engaged_user_surfacing_tweet_vectors.gap_294 * user_surfacing_tweet_vectors.gap_294) +
    (engaged_user_surfacing_tweet_vectors.gap_295 * user_surfacing_tweet_vectors.gap_295) +
    (engaged_user_surfacing_tweet_vectors.gap_296 * user_surfacing_tweet_vectors.gap_296) +
    (engaged_user_surfacing_tweet_vectors.gap_297 * user_surfacing_tweet_vectors.gap_297) +
    (engaged_user_surfacing_tweet_vectors.gap_298 * user_surfacing_tweet_vectors.gap_298) +
    (engaged_user_surfacing_tweet_vectors.gap_299 * user_surfacing_tweet_vectors.gap_299) +
    (engaged_user_surfacing_tweet_vectors.gap_300 * user_surfacing_tweet_vectors.gap_300) +
    (engaged_user_surfacing_tweet_vectors.gap_301 * user_surfacing_tweet_vectors.gap_301) +
    (engaged_user_surfacing_tweet_vectors.gap_302 * user_surfacing_tweet_vectors.gap_302) +
    (engaged_user_surfacing_tweet_vectors.gap_303 * user_surfacing_tweet_vectors.gap_303) +
    (engaged_user_surfacing_tweet_vectors.gap_304 * user_surfacing_tweet_vectors.gap_304) +
    (engaged_user_surfacing_tweet_vectors.gap_305 * user_surfacing_tweet_vectors.gap_305) +
    (engaged_user_surfacing_tweet_vectors.gap_306 * user_surfacing_tweet_vectors.gap_306) +
    (engaged_user_surfacing_tweet_vectors.gap_307 * user_surfacing_tweet_vectors.gap_307) +
    (engaged_user_surfacing_tweet_vectors.gap_308 * user_surfacing_tweet_vectors.gap_308) +
    (engaged_user_surfacing_tweet_vectors.gap_309 * user_surfacing_tweet_vectors.gap_309) +
    (engaged_user_surfacing_tweet_vectors.gap_310 * user_surfacing_tweet_vectors.gap_310) +
    (engaged_user_surfacing_tweet_vectors.gap_311 * user_surfacing_tweet_vectors.gap_311) +
    (engaged_user_surfacing_tweet_vectors.gap_312 * user_surfacing_tweet_vectors.gap_312) +
    (engaged_user_surfacing_tweet_vectors.gap_313 * user_surfacing_tweet_vectors.gap_313) +
    (engaged_user_surfacing_tweet_vectors.gap_314 * user_surfacing_tweet_vectors.gap_314) +
    (engaged_user_surfacing_tweet_vectors.gap_315 * user_surfacing_tweet_vectors.gap_315) +
    (engaged_user_surfacing_tweet_vectors.gap_316 * user_surfacing_tweet_vectors.gap_316) +
    (engaged_user_surfacing_tweet_vectors.gap_317 * user_surfacing_tweet_vectors.gap_317) +
    (engaged_user_surfacing_tweet_vectors.gap_318 * user_surfacing_tweet_vectors.gap_318) +
    (engaged_user_surfacing_tweet_vectors.gap_319 * user_surfacing_tweet_vectors.gap_319) +
    (engaged_user_surfacing_tweet_vectors.gap_320 * user_surfacing_tweet_vectors.gap_320) +
    (engaged_user_surfacing_tweet_vectors.gap_321 * user_surfacing_tweet_vectors.gap_321) +
    (engaged_user_surfacing_tweet_vectors.gap_322 * user_surfacing_tweet_vectors.gap_322) +
    (engaged_user_surfacing_tweet_vectors.gap_323 * user_surfacing_tweet_vectors.gap_323) +
    (engaged_user_surfacing_tweet_vectors.gap_324 * user_surfacing_tweet_vectors.gap_324) +
    (engaged_user_surfacing_tweet_vectors.gap_325 * user_surfacing_tweet_vectors.gap_325) +
    (engaged_user_surfacing_tweet_vectors.gap_326 * user_surfacing_tweet_vectors.gap_326) +
    (engaged_user_surfacing_tweet_vectors.gap_327 * user_surfacing_tweet_vectors.gap_327) +
    (engaged_user_surfacing_tweet_vectors.gap_328 * user_surfacing_tweet_vectors.gap_328) +
    (engaged_user_surfacing_tweet_vectors.gap_329 * user_surfacing_tweet_vectors.gap_329) +
    (engaged_user_surfacing_tweet_vectors.gap_330 * user_surfacing_tweet_vectors.gap_330) +
    (engaged_user_surfacing_tweet_vectors.gap_331 * user_surfacing_tweet_vectors.gap_331) +
    (engaged_user_surfacing_tweet_vectors.gap_332 * user_surfacing_tweet_vectors.gap_332) +
    (engaged_user_surfacing_tweet_vectors.gap_333 * user_surfacing_tweet_vectors.gap_333) +
    (engaged_user_surfacing_tweet_vectors.gap_334 * user_surfacing_tweet_vectors.gap_334) +
    (engaged_user_surfacing_tweet_vectors.gap_335 * user_surfacing_tweet_vectors.gap_335) +
    (engaged_user_surfacing_tweet_vectors.gap_336 * user_surfacing_tweet_vectors.gap_336) +
    (engaged_user_surfacing_tweet_vectors.gap_337 * user_surfacing_tweet_vectors.gap_337) +
    (engaged_user_surfacing_tweet_vectors.gap_338 * user_surfacing_tweet_vectors.gap_338) +
    (engaged_user_surfacing_tweet_vectors.gap_339 * user_surfacing_tweet_vectors.gap_339) +
    (engaged_user_surfacing_tweet_vectors.gap_340 * user_surfacing_tweet_vectors.gap_340) +
    (engaged_user_surfacing_tweet_vectors.gap_341 * user_surfacing_tweet_vectors.gap_341) +
    (engaged_user_surfacing_tweet_vectors.gap_342 * user_surfacing_tweet_vectors.gap_342) +
    (engaged_user_surfacing_tweet_vectors.gap_343 * user_surfacing_tweet_vectors.gap_343) +
    (engaged_user_surfacing_tweet_vectors.gap_344 * user_surfacing_tweet_vectors.gap_344) +
    (engaged_user_surfacing_tweet_vectors.gap_345 * user_surfacing_tweet_vectors.gap_345) +
    (engaged_user_surfacing_tweet_vectors.gap_346 * user_surfacing_tweet_vectors.gap_346) +
    (engaged_user_surfacing_tweet_vectors.gap_347 * user_surfacing_tweet_vectors.gap_347) +
    (engaged_user_surfacing_tweet_vectors.gap_348 * user_surfacing_tweet_vectors.gap_348) +
    (engaged_user_surfacing_tweet_vectors.gap_349 * user_surfacing_tweet_vectors.gap_349) +
    (engaged_user_surfacing_tweet_vectors.gap_350 * user_surfacing_tweet_vectors.gap_350) +
    (engaged_user_surfacing_tweet_vectors.gap_351 * user_surfacing_tweet_vectors.gap_351) +
    (engaged_user_surfacing_tweet_vectors.gap_352 * user_surfacing_tweet_vectors.gap_352) +
    (engaged_user_surfacing_tweet_vectors.gap_353 * user_surfacing_tweet_vectors.gap_353) +
    (engaged_user_surfacing_tweet_vectors.gap_354 * user_surfacing_tweet_vectors.gap_354) +
    (engaged_user_surfacing_tweet_vectors.gap_355 * user_surfacing_tweet_vectors.gap_355) +
    (engaged_user_surfacing_tweet_vectors.gap_356 * user_surfacing_tweet_vectors.gap_356) +
    (engaged_user_surfacing_tweet_vectors.gap_357 * user_surfacing_tweet_vectors.gap_357) +
    (engaged_user_surfacing_tweet_vectors.gap_358 * user_surfacing_tweet_vectors.gap_358) +
    (engaged_user_surfacing_tweet_vectors.gap_359 * user_surfacing_tweet_vectors.gap_359) +
    (engaged_user_surfacing_tweet_vectors.gap_360 * user_surfacing_tweet_vectors.gap_360) +
    (engaged_user_surfacing_tweet_vectors.gap_361 * user_surfacing_tweet_vectors.gap_361) +
    (engaged_user_surfacing_tweet_vectors.gap_362 * user_surfacing_tweet_vectors.gap_362) +
    (engaged_user_surfacing_tweet_vectors.gap_363 * user_surfacing_tweet_vectors.gap_363) +
    (engaged_user_surfacing_tweet_vectors.gap_364 * user_surfacing_tweet_vectors.gap_364) +
    (engaged_user_surfacing_tweet_vectors.gap_365 * user_surfacing_tweet_vectors.gap_365) +
    (engaged_user_surfacing_tweet_vectors.gap_366 * user_surfacing_tweet_vectors.gap_366) +
    (engaged_user_surfacing_tweet_vectors.gap_367 * user_surfacing_tweet_vectors.gap_367) +
    (engaged_user_surfacing_tweet_vectors.gap_368 * user_surfacing_tweet_vectors.gap_368) +
    (engaged_user_surfacing_tweet_vectors.gap_369 * user_surfacing_tweet_vectors.gap_369) +
    (engaged_user_surfacing_tweet_vectors.gap_370 * user_surfacing_tweet_vectors.gap_370) +
    (engaged_user_surfacing_tweet_vectors.gap_371 * user_surfacing_tweet_vectors.gap_371) +
    (engaged_user_surfacing_tweet_vectors.gap_372 * user_surfacing_tweet_vectors.gap_372) +
    (engaged_user_surfacing_tweet_vectors.gap_373 * user_surfacing_tweet_vectors.gap_373) +
    (engaged_user_surfacing_tweet_vectors.gap_374 * user_surfacing_tweet_vectors.gap_374) +
    (engaged_user_surfacing_tweet_vectors.gap_375 * user_surfacing_tweet_vectors.gap_375) +
    (engaged_user_surfacing_tweet_vectors.gap_376 * user_surfacing_tweet_vectors.gap_376) +
    (engaged_user_surfacing_tweet_vectors.gap_377 * user_surfacing_tweet_vectors.gap_377) +
    (engaged_user_surfacing_tweet_vectors.gap_378 * user_surfacing_tweet_vectors.gap_378) +
    (engaged_user_surfacing_tweet_vectors.gap_379 * user_surfacing_tweet_vectors.gap_379) +
    (engaged_user_surfacing_tweet_vectors.gap_380 * user_surfacing_tweet_vectors.gap_380) +
    (engaged_user_surfacing_tweet_vectors.gap_381 * user_surfacing_tweet_vectors.gap_381) +
    (engaged_user_surfacing_tweet_vectors.gap_382 * user_surfacing_tweet_vectors.gap_382) +
    (engaged_user_surfacing_tweet_vectors.gap_383 * user_surfacing_tweet_vectors.gap_383) +
    (engaged_user_surfacing_tweet_vectors.gap_384 * user_surfacing_tweet_vectors.gap_384) +
    (engaged_user_surfacing_tweet_vectors.gap_385 * user_surfacing_tweet_vectors.gap_385) +
    (engaged_user_surfacing_tweet_vectors.gap_386 * user_surfacing_tweet_vectors.gap_386) +
    (engaged_user_surfacing_tweet_vectors.gap_387 * user_surfacing_tweet_vectors.gap_387) +
    (engaged_user_surfacing_tweet_vectors.gap_388 * user_surfacing_tweet_vectors.gap_388) +
    (engaged_user_surfacing_tweet_vectors.gap_389 * user_surfacing_tweet_vectors.gap_389) +
    (engaged_user_surfacing_tweet_vectors.gap_390 * user_surfacing_tweet_vectors.gap_390) +
    (engaged_user_surfacing_tweet_vectors.gap_391 * user_surfacing_tweet_vectors.gap_391) +
    (engaged_user_surfacing_tweet_vectors.gap_392 * user_surfacing_tweet_vectors.gap_392) +
    (engaged_user_surfacing_tweet_vectors.gap_393 * user_surfacing_tweet_vectors.gap_393) +
    (engaged_user_surfacing_tweet_vectors.gap_394 * user_surfacing_tweet_vectors.gap_394) +
    (engaged_user_surfacing_tweet_vectors.gap_395 * user_surfacing_tweet_vectors.gap_395) +
    (engaged_user_surfacing_tweet_vectors.gap_396 * user_surfacing_tweet_vectors.gap_396) +
    (engaged_user_surfacing_tweet_vectors.gap_397 * user_surfacing_tweet_vectors.gap_397) +
    (engaged_user_surfacing_tweet_vectors.gap_398 * user_surfacing_tweet_vectors.gap_398) +
    (engaged_user_surfacing_tweet_vectors.gap_399 * user_surfacing_tweet_vectors.gap_399) +
    (engaged_user_surfacing_tweet_vectors.gap_400 * user_surfacing_tweet_vectors.gap_400) +
    (engaged_user_surfacing_tweet_vectors.gap_401 * user_surfacing_tweet_vectors.gap_401) +
    (engaged_user_surfacing_tweet_vectors.gap_402 * user_surfacing_tweet_vectors.gap_402) +
    (engaged_user_surfacing_tweet_vectors.gap_403 * user_surfacing_tweet_vectors.gap_403) +
    (engaged_user_surfacing_tweet_vectors.gap_404 * user_surfacing_tweet_vectors.gap_404) +
    (engaged_user_surfacing_tweet_vectors.gap_405 * user_surfacing_tweet_vectors.gap_405) +
    (engaged_user_surfacing_tweet_vectors.gap_406 * user_surfacing_tweet_vectors.gap_406) +
    (engaged_user_surfacing_tweet_vectors.gap_407 * user_surfacing_tweet_vectors.gap_407) +
    (engaged_user_surfacing_tweet_vectors.gap_408 * user_surfacing_tweet_vectors.gap_408) +
    (engaged_user_surfacing_tweet_vectors.gap_409 * user_surfacing_tweet_vectors.gap_409) +
    (engaged_user_surfacing_tweet_vectors.gap_410 * user_surfacing_tweet_vectors.gap_410) +
    (engaged_user_surfacing_tweet_vectors.gap_411 * user_surfacing_tweet_vectors.gap_411) +
    (engaged_user_surfacing_tweet_vectors.gap_412 * user_surfacing_tweet_vectors.gap_412) +
    (engaged_user_surfacing_tweet_vectors.gap_413 * user_surfacing_tweet_vectors.gap_413) +
    (engaged_user_surfacing_tweet_vectors.gap_414 * user_surfacing_tweet_vectors.gap_414) +
    (engaged_user_surfacing_tweet_vectors.gap_415 * user_surfacing_tweet_vectors.gap_415) +
    (engaged_user_surfacing_tweet_vectors.gap_416 * user_surfacing_tweet_vectors.gap_416) +
    (engaged_user_surfacing_tweet_vectors.gap_417 * user_surfacing_tweet_vectors.gap_417) +
    (engaged_user_surfacing_tweet_vectors.gap_418 * user_surfacing_tweet_vectors.gap_418) +
    (engaged_user_surfacing_tweet_vectors.gap_419 * user_surfacing_tweet_vectors.gap_419) +
    (engaged_user_surfacing_tweet_vectors.gap_420 * user_surfacing_tweet_vectors.gap_420) +
    (engaged_user_surfacing_tweet_vectors.gap_421 * user_surfacing_tweet_vectors.gap_421) +
    (engaged_user_surfacing_tweet_vectors.gap_422 * user_surfacing_tweet_vectors.gap_422) +
    (engaged_user_surfacing_tweet_vectors.gap_423 * user_surfacing_tweet_vectors.gap_423) +
    (engaged_user_surfacing_tweet_vectors.gap_424 * user_surfacing_tweet_vectors.gap_424) +
    (engaged_user_surfacing_tweet_vectors.gap_425 * user_surfacing_tweet_vectors.gap_425) +
    (engaged_user_surfacing_tweet_vectors.gap_426 * user_surfacing_tweet_vectors.gap_426) +
    (engaged_user_surfacing_tweet_vectors.gap_427 * user_surfacing_tweet_vectors.gap_427) +
    (engaged_user_surfacing_tweet_vectors.gap_428 * user_surfacing_tweet_vectors.gap_428) +
    (engaged_user_surfacing_tweet_vectors.gap_429 * user_surfacing_tweet_vectors.gap_429) +
    (engaged_user_surfacing_tweet_vectors.gap_430 * user_surfacing_tweet_vectors.gap_430) +
    (engaged_user_surfacing_tweet_vectors.gap_431 * user_surfacing_tweet_vectors.gap_431) +
    (engaged_user_surfacing_tweet_vectors.gap_432 * user_surfacing_tweet_vectors.gap_432) +
    (engaged_user_surfacing_tweet_vectors.gap_433 * user_surfacing_tweet_vectors.gap_433) +
    (engaged_user_surfacing_tweet_vectors.gap_434 * user_surfacing_tweet_vectors.gap_434) +
    (engaged_user_surfacing_tweet_vectors.gap_435 * user_surfacing_tweet_vectors.gap_435) +
    (engaged_user_surfacing_tweet_vectors.gap_436 * user_surfacing_tweet_vectors.gap_436) +
    (engaged_user_surfacing_tweet_vectors.gap_437 * user_surfacing_tweet_vectors.gap_437) +
    (engaged_user_surfacing_tweet_vectors.gap_438 * user_surfacing_tweet_vectors.gap_438) +
    (engaged_user_surfacing_tweet_vectors.gap_439 * user_surfacing_tweet_vectors.gap_439) +
    (engaged_user_surfacing_tweet_vectors.gap_440 * user_surfacing_tweet_vectors.gap_440) +
    (engaged_user_surfacing_tweet_vectors.gap_441 * user_surfacing_tweet_vectors.gap_441) +
    (engaged_user_surfacing_tweet_vectors.gap_442 * user_surfacing_tweet_vectors.gap_442) +
    (engaged_user_surfacing_tweet_vectors.gap_443 * user_surfacing_tweet_vectors.gap_443) +
    (engaged_user_surfacing_tweet_vectors.gap_444 * user_surfacing_tweet_vectors.gap_444) +
    (engaged_user_surfacing_tweet_vectors.gap_445 * user_surfacing_tweet_vectors.gap_445) +
    (engaged_user_surfacing_tweet_vectors.gap_446 * user_surfacing_tweet_vectors.gap_446) +
    (engaged_user_surfacing_tweet_vectors.gap_447 * user_surfacing_tweet_vectors.gap_447) +
    (engaged_user_surfacing_tweet_vectors.gap_448 * user_surfacing_tweet_vectors.gap_448) +
    (engaged_user_surfacing_tweet_vectors.gap_449 * user_surfacing_tweet_vectors.gap_449) +
    (engaged_user_surfacing_tweet_vectors.gap_450 * user_surfacing_tweet_vectors.gap_450) +
    (engaged_user_surfacing_tweet_vectors.gap_451 * user_surfacing_tweet_vectors.gap_451) +
    (engaged_user_surfacing_tweet_vectors.gap_452 * user_surfacing_tweet_vectors.gap_452) +
    (engaged_user_surfacing_tweet_vectors.gap_453 * user_surfacing_tweet_vectors.gap_453) +
    (engaged_user_surfacing_tweet_vectors.gap_454 * user_surfacing_tweet_vectors.gap_454) +
    (engaged_user_surfacing_tweet_vectors.gap_455 * user_surfacing_tweet_vectors.gap_455) +
    (engaged_user_surfacing_tweet_vectors.gap_456 * user_surfacing_tweet_vectors.gap_456) +
    (engaged_user_surfacing_tweet_vectors.gap_457 * user_surfacing_tweet_vectors.gap_457) +
    (engaged_user_surfacing_tweet_vectors.gap_458 * user_surfacing_tweet_vectors.gap_458) +
    (engaged_user_surfacing_tweet_vectors.gap_459 * user_surfacing_tweet_vectors.gap_459) +
    (engaged_user_surfacing_tweet_vectors.gap_460 * user_surfacing_tweet_vectors.gap_460) +
    (engaged_user_surfacing_tweet_vectors.gap_461 * user_surfacing_tweet_vectors.gap_461) +
    (engaged_user_surfacing_tweet_vectors.gap_462 * user_surfacing_tweet_vectors.gap_462) +
    (engaged_user_surfacing_tweet_vectors.gap_463 * user_surfacing_tweet_vectors.gap_463) +
    (engaged_user_surfacing_tweet_vectors.gap_464 * user_surfacing_tweet_vectors.gap_464) +
    (engaged_user_surfacing_tweet_vectors.gap_465 * user_surfacing_tweet_vectors.gap_465) +
    (engaged_user_surfacing_tweet_vectors.gap_466 * user_surfacing_tweet_vectors.gap_466) +
    (engaged_user_surfacing_tweet_vectors.gap_467 * user_surfacing_tweet_vectors.gap_467) +
    (engaged_user_surfacing_tweet_vectors.gap_468 * user_surfacing_tweet_vectors.gap_468) +
    (engaged_user_surfacing_tweet_vectors.gap_469 * user_surfacing_tweet_vectors.gap_469) +
    (engaged_user_surfacing_tweet_vectors.gap_470 * user_surfacing_tweet_vectors.gap_470) +
    (engaged_user_surfacing_tweet_vectors.gap_471 * user_surfacing_tweet_vectors.gap_471) +
    (engaged_user_surfacing_tweet_vectors.gap_472 * user_surfacing_tweet_vectors.gap_472) +
    (engaged_user_surfacing_tweet_vectors.gap_473 * user_surfacing_tweet_vectors.gap_473) +
    (engaged_user_surfacing_tweet_vectors.gap_474 * user_surfacing_tweet_vectors.gap_474) +
    (engaged_user_surfacing_tweet_vectors.gap_475 * user_surfacing_tweet_vectors.gap_475) +
    (engaged_user_surfacing_tweet_vectors.gap_476 * user_surfacing_tweet_vectors.gap_476) +
    (engaged_user_surfacing_tweet_vectors.gap_477 * user_surfacing_tweet_vectors.gap_477) +
    (engaged_user_surfacing_tweet_vectors.gap_478 * user_surfacing_tweet_vectors.gap_478) +
    (engaged_user_surfacing_tweet_vectors.gap_479 * user_surfacing_tweet_vectors.gap_479) +
    (engaged_user_surfacing_tweet_vectors.gap_480 * user_surfacing_tweet_vectors.gap_480) +
    (engaged_user_surfacing_tweet_vectors.gap_481 * user_surfacing_tweet_vectors.gap_481) +
    (engaged_user_surfacing_tweet_vectors.gap_482 * user_surfacing_tweet_vectors.gap_482) +
    (engaged_user_surfacing_tweet_vectors.gap_483 * user_surfacing_tweet_vectors.gap_483) +
    (engaged_user_surfacing_tweet_vectors.gap_484 * user_surfacing_tweet_vectors.gap_484) +
    (engaged_user_surfacing_tweet_vectors.gap_485 * user_surfacing_tweet_vectors.gap_485) +
    (engaged_user_surfacing_tweet_vectors.gap_486 * user_surfacing_tweet_vectors.gap_486) +
    (engaged_user_surfacing_tweet_vectors.gap_487 * user_surfacing_tweet_vectors.gap_487) +
    (engaged_user_surfacing_tweet_vectors.gap_488 * user_surfacing_tweet_vectors.gap_488) +
    (engaged_user_surfacing_tweet_vectors.gap_489 * user_surfacing_tweet_vectors.gap_489) +
    (engaged_user_surfacing_tweet_vectors.gap_490 * user_surfacing_tweet_vectors.gap_490) +
    (engaged_user_surfacing_tweet_vectors.gap_491 * user_surfacing_tweet_vectors.gap_491) +
    (engaged_user_surfacing_tweet_vectors.gap_492 * user_surfacing_tweet_vectors.gap_492) +
    (engaged_user_surfacing_tweet_vectors.gap_493 * user_surfacing_tweet_vectors.gap_493) +
    (engaged_user_surfacing_tweet_vectors.gap_494 * user_surfacing_tweet_vectors.gap_494) +
    (engaged_user_surfacing_tweet_vectors.gap_495 * user_surfacing_tweet_vectors.gap_495) +
    (engaged_user_surfacing_tweet_vectors.gap_496 * user_surfacing_tweet_vectors.gap_496) +
    (engaged_user_surfacing_tweet_vectors.gap_497 * user_surfacing_tweet_vectors.gap_497) +
    (engaged_user_surfacing_tweet_vectors.gap_498 * user_surfacing_tweet_vectors.gap_498) +
    (engaged_user_surfacing_tweet_vectors.gap_499 * user_surfacing_tweet_vectors.gap_499) +
    (engaged_user_surfacing_tweet_vectors.gap_500 * user_surfacing_tweet_vectors.gap_500) +
    (engaged_user_surfacing_tweet_vectors.gap_501 * user_surfacing_tweet_vectors.gap_501) +
    (engaged_user_surfacing_tweet_vectors.gap_502 * user_surfacing_tweet_vectors.gap_502) +
    (engaged_user_surfacing_tweet_vectors.gap_503 * user_surfacing_tweet_vectors.gap_503) +
    (engaged_user_surfacing_tweet_vectors.gap_504 * user_surfacing_tweet_vectors.gap_504) +
    (engaged_user_surfacing_tweet_vectors.gap_505 * user_surfacing_tweet_vectors.gap_505) +
    (engaged_user_surfacing_tweet_vectors.gap_506 * user_surfacing_tweet_vectors.gap_506) +
    (engaged_user_surfacing_tweet_vectors.gap_507 * user_surfacing_tweet_vectors.gap_507) +
    (engaged_user_surfacing_tweet_vectors.gap_508 * user_surfacing_tweet_vectors.gap_508) +
    (engaged_user_surfacing_tweet_vectors.gap_509 * user_surfacing_tweet_vectors.gap_509) +
    (engaged_user_surfacing_tweet_vectors.gap_510 * user_surfacing_tweet_vectors.gap_510) +
    (engaged_user_surfacing_tweet_vectors.gap_511 * user_surfacing_tweet_vectors.gap_511) +
    (engaged_user_surfacing_tweet_vectors.gap_512 * user_surfacing_tweet_vectors.gap_512) +
    (engaged_user_surfacing_tweet_vectors.gap_513 * user_surfacing_tweet_vectors.gap_513) +
    (engaged_user_surfacing_tweet_vectors.gap_514 * user_surfacing_tweet_vectors.gap_514) +
    (engaged_user_surfacing_tweet_vectors.gap_515 * user_surfacing_tweet_vectors.gap_515) +
    (engaged_user_surfacing_tweet_vectors.gap_516 * user_surfacing_tweet_vectors.gap_516) +
    (engaged_user_surfacing_tweet_vectors.gap_517 * user_surfacing_tweet_vectors.gap_517) +
    (engaged_user_surfacing_tweet_vectors.gap_518 * user_surfacing_tweet_vectors.gap_518) +
    (engaged_user_surfacing_tweet_vectors.gap_519 * user_surfacing_tweet_vectors.gap_519) +
    (engaged_user_surfacing_tweet_vectors.gap_520 * user_surfacing_tweet_vectors.gap_520) +
    (engaged_user_surfacing_tweet_vectors.gap_521 * user_surfacing_tweet_vectors.gap_521) +
    (engaged_user_surfacing_tweet_vectors.gap_522 * user_surfacing_tweet_vectors.gap_522) +
    (engaged_user_surfacing_tweet_vectors.gap_523 * user_surfacing_tweet_vectors.gap_523) +
    (engaged_user_surfacing_tweet_vectors.gap_524 * user_surfacing_tweet_vectors.gap_524) +
    (engaged_user_surfacing_tweet_vectors.gap_525 * user_surfacing_tweet_vectors.gap_525) +
    (engaged_user_surfacing_tweet_vectors.gap_526 * user_surfacing_tweet_vectors.gap_526) +
    (engaged_user_surfacing_tweet_vectors.gap_527 * user_surfacing_tweet_vectors.gap_527) +
    (engaged_user_surfacing_tweet_vectors.gap_528 * user_surfacing_tweet_vectors.gap_528) +
    (engaged_user_surfacing_tweet_vectors.gap_529 * user_surfacing_tweet_vectors.gap_529) +
    (engaged_user_surfacing_tweet_vectors.gap_530 * user_surfacing_tweet_vectors.gap_530) +
    (engaged_user_surfacing_tweet_vectors.gap_531 * user_surfacing_tweet_vectors.gap_531) +
    (engaged_user_surfacing_tweet_vectors.gap_532 * user_surfacing_tweet_vectors.gap_532) +
    (engaged_user_surfacing_tweet_vectors.gap_533 * user_surfacing_tweet_vectors.gap_533) +
    (engaged_user_surfacing_tweet_vectors.gap_534 * user_surfacing_tweet_vectors.gap_534) +
    (engaged_user_surfacing_tweet_vectors.gap_535 * user_surfacing_tweet_vectors.gap_535) +
    (engaged_user_surfacing_tweet_vectors.gap_536 * user_surfacing_tweet_vectors.gap_536) +
    (engaged_user_surfacing_tweet_vectors.gap_537 * user_surfacing_tweet_vectors.gap_537) +
    (engaged_user_surfacing_tweet_vectors.gap_538 * user_surfacing_tweet_vectors.gap_538) +
    (engaged_user_surfacing_tweet_vectors.gap_539 * user_surfacing_tweet_vectors.gap_539) +
    (engaged_user_surfacing_tweet_vectors.gap_540 * user_surfacing_tweet_vectors.gap_540) +
    (engaged_user_surfacing_tweet_vectors.gap_541 * user_surfacing_tweet_vectors.gap_541) +
    (engaged_user_surfacing_tweet_vectors.gap_542 * user_surfacing_tweet_vectors.gap_542) +
    (engaged_user_surfacing_tweet_vectors.gap_543 * user_surfacing_tweet_vectors.gap_543) +
    (engaged_user_surfacing_tweet_vectors.gap_544 * user_surfacing_tweet_vectors.gap_544) +
    (engaged_user_surfacing_tweet_vectors.gap_545 * user_surfacing_tweet_vectors.gap_545) +
    (engaged_user_surfacing_tweet_vectors.gap_546 * user_surfacing_tweet_vectors.gap_546) +
    (engaged_user_surfacing_tweet_vectors.gap_547 * user_surfacing_tweet_vectors.gap_547) +
    (engaged_user_surfacing_tweet_vectors.gap_548 * user_surfacing_tweet_vectors.gap_548) +
    (engaged_user_surfacing_tweet_vectors.gap_549 * user_surfacing_tweet_vectors.gap_549) +
    (engaged_user_surfacing_tweet_vectors.gap_550 * user_surfacing_tweet_vectors.gap_550) +
    (engaged_user_surfacing_tweet_vectors.gap_551 * user_surfacing_tweet_vectors.gap_551) +
    (engaged_user_surfacing_tweet_vectors.gap_552 * user_surfacing_tweet_vectors.gap_552) +
    (engaged_user_surfacing_tweet_vectors.gap_553 * user_surfacing_tweet_vectors.gap_553) +
    (engaged_user_surfacing_tweet_vectors.gap_554 * user_surfacing_tweet_vectors.gap_554) +
    (engaged_user_surfacing_tweet_vectors.gap_555 * user_surfacing_tweet_vectors.gap_555) +
    (engaged_user_surfacing_tweet_vectors.gap_556 * user_surfacing_tweet_vectors.gap_556) +
    (engaged_user_surfacing_tweet_vectors.gap_557 * user_surfacing_tweet_vectors.gap_557) +
    (engaged_user_surfacing_tweet_vectors.gap_558 * user_surfacing_tweet_vectors.gap_558) +
    (engaged_user_surfacing_tweet_vectors.gap_559 * user_surfacing_tweet_vectors.gap_559) +
    (engaged_user_surfacing_tweet_vectors.gap_560 * user_surfacing_tweet_vectors.gap_560) +
    (engaged_user_surfacing_tweet_vectors.gap_561 * user_surfacing_tweet_vectors.gap_561) +
    (engaged_user_surfacing_tweet_vectors.gap_562 * user_surfacing_tweet_vectors.gap_562) +
    (engaged_user_surfacing_tweet_vectors.gap_563 * user_surfacing_tweet_vectors.gap_563) +
    (engaged_user_surfacing_tweet_vectors.gap_564 * user_surfacing_tweet_vectors.gap_564) +
    (engaged_user_surfacing_tweet_vectors.gap_565 * user_surfacing_tweet_vectors.gap_565) +
    (engaged_user_surfacing_tweet_vectors.gap_566 * user_surfacing_tweet_vectors.gap_566) +
    (engaged_user_surfacing_tweet_vectors.gap_567 * user_surfacing_tweet_vectors.gap_567) +
    (engaged_user_surfacing_tweet_vectors.gap_568 * user_surfacing_tweet_vectors.gap_568) +
    (engaged_user_surfacing_tweet_vectors.gap_569 * user_surfacing_tweet_vectors.gap_569) +
    (engaged_user_surfacing_tweet_vectors.gap_570 * user_surfacing_tweet_vectors.gap_570) +
    (engaged_user_surfacing_tweet_vectors.gap_571 * user_surfacing_tweet_vectors.gap_571) +
    (engaged_user_surfacing_tweet_vectors.gap_572 * user_surfacing_tweet_vectors.gap_572) +
    (engaged_user_surfacing_tweet_vectors.gap_573 * user_surfacing_tweet_vectors.gap_573) +
    (engaged_user_surfacing_tweet_vectors.gap_574 * user_surfacing_tweet_vectors.gap_574) +
    (engaged_user_surfacing_tweet_vectors.gap_575 * user_surfacing_tweet_vectors.gap_575) +
    (engaged_user_surfacing_tweet_vectors.gap_576 * user_surfacing_tweet_vectors.gap_576) +
    (engaged_user_surfacing_tweet_vectors.gap_577 * user_surfacing_tweet_vectors.gap_577) +
    (engaged_user_surfacing_tweet_vectors.gap_578 * user_surfacing_tweet_vectors.gap_578) +
    (engaged_user_surfacing_tweet_vectors.gap_579 * user_surfacing_tweet_vectors.gap_579) +
    (engaged_user_surfacing_tweet_vectors.gap_580 * user_surfacing_tweet_vectors.gap_580) +
    (engaged_user_surfacing_tweet_vectors.gap_581 * user_surfacing_tweet_vectors.gap_581) +
    (engaged_user_surfacing_tweet_vectors.gap_582 * user_surfacing_tweet_vectors.gap_582) +
    (engaged_user_surfacing_tweet_vectors.gap_583 * user_surfacing_tweet_vectors.gap_583) +
    (engaged_user_surfacing_tweet_vectors.gap_584 * user_surfacing_tweet_vectors.gap_584) +
    (engaged_user_surfacing_tweet_vectors.gap_585 * user_surfacing_tweet_vectors.gap_585) +
    (engaged_user_surfacing_tweet_vectors.gap_586 * user_surfacing_tweet_vectors.gap_586) +
    (engaged_user_surfacing_tweet_vectors.gap_587 * user_surfacing_tweet_vectors.gap_587) +
    (engaged_user_surfacing_tweet_vectors.gap_588 * user_surfacing_tweet_vectors.gap_588) +
    (engaged_user_surfacing_tweet_vectors.gap_589 * user_surfacing_tweet_vectors.gap_589) +
    (engaged_user_surfacing_tweet_vectors.gap_590 * user_surfacing_tweet_vectors.gap_590) +
    (engaged_user_surfacing_tweet_vectors.gap_591 * user_surfacing_tweet_vectors.gap_591) +
    (engaged_user_surfacing_tweet_vectors.gap_592 * user_surfacing_tweet_vectors.gap_592) +
    (engaged_user_surfacing_tweet_vectors.gap_593 * user_surfacing_tweet_vectors.gap_593) +
    (engaged_user_surfacing_tweet_vectors.gap_594 * user_surfacing_tweet_vectors.gap_594) +
    (engaged_user_surfacing_tweet_vectors.gap_595 * user_surfacing_tweet_vectors.gap_595) +
    (engaged_user_surfacing_tweet_vectors.gap_596 * user_surfacing_tweet_vectors.gap_596) +
    (engaged_user_surfacing_tweet_vectors.gap_597 * user_surfacing_tweet_vectors.gap_597) +
    (engaged_user_surfacing_tweet_vectors.gap_598 * user_surfacing_tweet_vectors.gap_598) +
    (engaged_user_surfacing_tweet_vectors.gap_599 * user_surfacing_tweet_vectors.gap_599) +
    (engaged_user_surfacing_tweet_vectors.gap_600 * user_surfacing_tweet_vectors.gap_600) +
    (engaged_user_surfacing_tweet_vectors.gap_601 * user_surfacing_tweet_vectors.gap_601) +
    (engaged_user_surfacing_tweet_vectors.gap_602 * user_surfacing_tweet_vectors.gap_602) +
    (engaged_user_surfacing_tweet_vectors.gap_603 * user_surfacing_tweet_vectors.gap_603) +
    (engaged_user_surfacing_tweet_vectors.gap_604 * user_surfacing_tweet_vectors.gap_604) +
    (engaged_user_surfacing_tweet_vectors.gap_605 * user_surfacing_tweet_vectors.gap_605) +
    (engaged_user_surfacing_tweet_vectors.gap_606 * user_surfacing_tweet_vectors.gap_606) +
    (engaged_user_surfacing_tweet_vectors.gap_607 * user_surfacing_tweet_vectors.gap_607) +
    (engaged_user_surfacing_tweet_vectors.gap_608 * user_surfacing_tweet_vectors.gap_608) +
    (engaged_user_surfacing_tweet_vectors.gap_609 * user_surfacing_tweet_vectors.gap_609) +
    (engaged_user_surfacing_tweet_vectors.gap_610 * user_surfacing_tweet_vectors.gap_610) +
    (engaged_user_surfacing_tweet_vectors.gap_611 * user_surfacing_tweet_vectors.gap_611) +
    (engaged_user_surfacing_tweet_vectors.gap_612 * user_surfacing_tweet_vectors.gap_612) +
    (engaged_user_surfacing_tweet_vectors.gap_613 * user_surfacing_tweet_vectors.gap_613) +
    (engaged_user_surfacing_tweet_vectors.gap_614 * user_surfacing_tweet_vectors.gap_614) +
    (engaged_user_surfacing_tweet_vectors.gap_615 * user_surfacing_tweet_vectors.gap_615) +
    (engaged_user_surfacing_tweet_vectors.gap_616 * user_surfacing_tweet_vectors.gap_616) +
    (engaged_user_surfacing_tweet_vectors.gap_617 * user_surfacing_tweet_vectors.gap_617) +
    (engaged_user_surfacing_tweet_vectors.gap_618 * user_surfacing_tweet_vectors.gap_618) +
    (engaged_user_surfacing_tweet_vectors.gap_619 * user_surfacing_tweet_vectors.gap_619) +
    (engaged_user_surfacing_tweet_vectors.gap_620 * user_surfacing_tweet_vectors.gap_620) +
    (engaged_user_surfacing_tweet_vectors.gap_621 * user_surfacing_tweet_vectors.gap_621) +
    (engaged_user_surfacing_tweet_vectors.gap_622 * user_surfacing_tweet_vectors.gap_622) +
    (engaged_user_surfacing_tweet_vectors.gap_623 * user_surfacing_tweet_vectors.gap_623) +
    (engaged_user_surfacing_tweet_vectors.gap_624 * user_surfacing_tweet_vectors.gap_624) +
    (engaged_user_surfacing_tweet_vectors.gap_625 * user_surfacing_tweet_vectors.gap_625) +
    (engaged_user_surfacing_tweet_vectors.gap_626 * user_surfacing_tweet_vectors.gap_626) +
    (engaged_user_surfacing_tweet_vectors.gap_627 * user_surfacing_tweet_vectors.gap_627) +
    (engaged_user_surfacing_tweet_vectors.gap_628 * user_surfacing_tweet_vectors.gap_628) +
    (engaged_user_surfacing_tweet_vectors.gap_629 * user_surfacing_tweet_vectors.gap_629) +
    (engaged_user_surfacing_tweet_vectors.gap_630 * user_surfacing_tweet_vectors.gap_630) +
    (engaged_user_surfacing_tweet_vectors.gap_631 * user_surfacing_tweet_vectors.gap_631) +
    (engaged_user_surfacing_tweet_vectors.gap_632 * user_surfacing_tweet_vectors.gap_632) +
    (engaged_user_surfacing_tweet_vectors.gap_633 * user_surfacing_tweet_vectors.gap_633) +
    (engaged_user_surfacing_tweet_vectors.gap_634 * user_surfacing_tweet_vectors.gap_634) +
    (engaged_user_surfacing_tweet_vectors.gap_635 * user_surfacing_tweet_vectors.gap_635) +
    (engaged_user_surfacing_tweet_vectors.gap_636 * user_surfacing_tweet_vectors.gap_636) +
    (engaged_user_surfacing_tweet_vectors.gap_637 * user_surfacing_tweet_vectors.gap_637) +
    (engaged_user_surfacing_tweet_vectors.gap_638 * user_surfacing_tweet_vectors.gap_638) +
    (engaged_user_surfacing_tweet_vectors.gap_639 * user_surfacing_tweet_vectors.gap_639) +
    (engaged_user_surfacing_tweet_vectors.gap_640 * user_surfacing_tweet_vectors.gap_640) +
    (engaged_user_surfacing_tweet_vectors.gap_641 * user_surfacing_tweet_vectors.gap_641) +
    (engaged_user_surfacing_tweet_vectors.gap_642 * user_surfacing_tweet_vectors.gap_642) +
    (engaged_user_surfacing_tweet_vectors.gap_643 * user_surfacing_tweet_vectors.gap_643) +
    (engaged_user_surfacing_tweet_vectors.gap_644 * user_surfacing_tweet_vectors.gap_644) +
    (engaged_user_surfacing_tweet_vectors.gap_645 * user_surfacing_tweet_vectors.gap_645) +
    (engaged_user_surfacing_tweet_vectors.gap_646 * user_surfacing_tweet_vectors.gap_646) +
    (engaged_user_surfacing_tweet_vectors.gap_647 * user_surfacing_tweet_vectors.gap_647) +
    (engaged_user_surfacing_tweet_vectors.gap_648 * user_surfacing_tweet_vectors.gap_648) +
    (engaged_user_surfacing_tweet_vectors.gap_649 * user_surfacing_tweet_vectors.gap_649) +
    (engaged_user_surfacing_tweet_vectors.gap_650 * user_surfacing_tweet_vectors.gap_650) +
    (engaged_user_surfacing_tweet_vectors.gap_651 * user_surfacing_tweet_vectors.gap_651) +
    (engaged_user_surfacing_tweet_vectors.gap_652 * user_surfacing_tweet_vectors.gap_652) +
    (engaged_user_surfacing_tweet_vectors.gap_653 * user_surfacing_tweet_vectors.gap_653) +
    (engaged_user_surfacing_tweet_vectors.gap_654 * user_surfacing_tweet_vectors.gap_654) +
    (engaged_user_surfacing_tweet_vectors.gap_655 * user_surfacing_tweet_vectors.gap_655) +
    (engaged_user_surfacing_tweet_vectors.gap_656 * user_surfacing_tweet_vectors.gap_656) +
    (engaged_user_surfacing_tweet_vectors.gap_657 * user_surfacing_tweet_vectors.gap_657) +
    (engaged_user_surfacing_tweet_vectors.gap_658 * user_surfacing_tweet_vectors.gap_658) +
    (engaged_user_surfacing_tweet_vectors.gap_659 * user_surfacing_tweet_vectors.gap_659) +
    (engaged_user_surfacing_tweet_vectors.gap_660 * user_surfacing_tweet_vectors.gap_660) +
    (engaged_user_surfacing_tweet_vectors.gap_661 * user_surfacing_tweet_vectors.gap_661) +
    (engaged_user_surfacing_tweet_vectors.gap_662 * user_surfacing_tweet_vectors.gap_662) +
    (engaged_user_surfacing_tweet_vectors.gap_663 * user_surfacing_tweet_vectors.gap_663) +
    (engaged_user_surfacing_tweet_vectors.gap_664 * user_surfacing_tweet_vectors.gap_664) +
    (engaged_user_surfacing_tweet_vectors.gap_665 * user_surfacing_tweet_vectors.gap_665) +
    (engaged_user_surfacing_tweet_vectors.gap_666 * user_surfacing_tweet_vectors.gap_666) +
    (engaged_user_surfacing_tweet_vectors.gap_667 * user_surfacing_tweet_vectors.gap_667) +
    (engaged_user_surfacing_tweet_vectors.gap_668 * user_surfacing_tweet_vectors.gap_668) +
    (engaged_user_surfacing_tweet_vectors.gap_669 * user_surfacing_tweet_vectors.gap_669) +
    (engaged_user_surfacing_tweet_vectors.gap_670 * user_surfacing_tweet_vectors.gap_670) +
    (engaged_user_surfacing_tweet_vectors.gap_671 * user_surfacing_tweet_vectors.gap_671) +
    (engaged_user_surfacing_tweet_vectors.gap_672 * user_surfacing_tweet_vectors.gap_672) +
    (engaged_user_surfacing_tweet_vectors.gap_673 * user_surfacing_tweet_vectors.gap_673) +
    (engaged_user_surfacing_tweet_vectors.gap_674 * user_surfacing_tweet_vectors.gap_674) +
    (engaged_user_surfacing_tweet_vectors.gap_675 * user_surfacing_tweet_vectors.gap_675) +
    (engaged_user_surfacing_tweet_vectors.gap_676 * user_surfacing_tweet_vectors.gap_676) +
    (engaged_user_surfacing_tweet_vectors.gap_677 * user_surfacing_tweet_vectors.gap_677) +
    (engaged_user_surfacing_tweet_vectors.gap_678 * user_surfacing_tweet_vectors.gap_678) +
    (engaged_user_surfacing_tweet_vectors.gap_679 * user_surfacing_tweet_vectors.gap_679) +
    (engaged_user_surfacing_tweet_vectors.gap_680 * user_surfacing_tweet_vectors.gap_680) +
    (engaged_user_surfacing_tweet_vectors.gap_681 * user_surfacing_tweet_vectors.gap_681) +
    (engaged_user_surfacing_tweet_vectors.gap_682 * user_surfacing_tweet_vectors.gap_682) +
    (engaged_user_surfacing_tweet_vectors.gap_683 * user_surfacing_tweet_vectors.gap_683) +
    (engaged_user_surfacing_tweet_vectors.gap_684 * user_surfacing_tweet_vectors.gap_684) +
    (engaged_user_surfacing_tweet_vectors.gap_685 * user_surfacing_tweet_vectors.gap_685) +
    (engaged_user_surfacing_tweet_vectors.gap_686 * user_surfacing_tweet_vectors.gap_686) +
    (engaged_user_surfacing_tweet_vectors.gap_687 * user_surfacing_tweet_vectors.gap_687) +
    (engaged_user_surfacing_tweet_vectors.gap_688 * user_surfacing_tweet_vectors.gap_688) +
    (engaged_user_surfacing_tweet_vectors.gap_689 * user_surfacing_tweet_vectors.gap_689) +
    (engaged_user_surfacing_tweet_vectors.gap_690 * user_surfacing_tweet_vectors.gap_690) +
    (engaged_user_surfacing_tweet_vectors.gap_691 * user_surfacing_tweet_vectors.gap_691) +
    (engaged_user_surfacing_tweet_vectors.gap_692 * user_surfacing_tweet_vectors.gap_692) +
    (engaged_user_surfacing_tweet_vectors.gap_693 * user_surfacing_tweet_vectors.gap_693) +
    (engaged_user_surfacing_tweet_vectors.gap_694 * user_surfacing_tweet_vectors.gap_694) +
    (engaged_user_surfacing_tweet_vectors.gap_695 * user_surfacing_tweet_vectors.gap_695) +
    (engaged_user_surfacing_tweet_vectors.gap_696 * user_surfacing_tweet_vectors.gap_696) +
    (engaged_user_surfacing_tweet_vectors.gap_697 * user_surfacing_tweet_vectors.gap_697) +
    (engaged_user_surfacing_tweet_vectors.gap_698 * user_surfacing_tweet_vectors.gap_698) +
    (engaged_user_surfacing_tweet_vectors.gap_699 * user_surfacing_tweet_vectors.gap_699) +
    (engaged_user_surfacing_tweet_vectors.gap_700 * user_surfacing_tweet_vectors.gap_700) +
    (engaged_user_surfacing_tweet_vectors.gap_701 * user_surfacing_tweet_vectors.gap_701) +
    (engaged_user_surfacing_tweet_vectors.gap_702 * user_surfacing_tweet_vectors.gap_702) +
    (engaged_user_surfacing_tweet_vectors.gap_703 * user_surfacing_tweet_vectors.gap_703) +
    (engaged_user_surfacing_tweet_vectors.gap_704 * user_surfacing_tweet_vectors.gap_704) +
    (engaged_user_surfacing_tweet_vectors.gap_705 * user_surfacing_tweet_vectors.gap_705) +
    (engaged_user_surfacing_tweet_vectors.gap_706 * user_surfacing_tweet_vectors.gap_706) +
    (engaged_user_surfacing_tweet_vectors.gap_707 * user_surfacing_tweet_vectors.gap_707) +
    (engaged_user_surfacing_tweet_vectors.gap_708 * user_surfacing_tweet_vectors.gap_708) +
    (engaged_user_surfacing_tweet_vectors.gap_709 * user_surfacing_tweet_vectors.gap_709) +
    (engaged_user_surfacing_tweet_vectors.gap_710 * user_surfacing_tweet_vectors.gap_710) +
    (engaged_user_surfacing_tweet_vectors.gap_711 * user_surfacing_tweet_vectors.gap_711) +
    (engaged_user_surfacing_tweet_vectors.gap_712 * user_surfacing_tweet_vectors.gap_712) +
    (engaged_user_surfacing_tweet_vectors.gap_713 * user_surfacing_tweet_vectors.gap_713) +
    (engaged_user_surfacing_tweet_vectors.gap_714 * user_surfacing_tweet_vectors.gap_714) +
    (engaged_user_surfacing_tweet_vectors.gap_715 * user_surfacing_tweet_vectors.gap_715) +
    (engaged_user_surfacing_tweet_vectors.gap_716 * user_surfacing_tweet_vectors.gap_716) +
    (engaged_user_surfacing_tweet_vectors.gap_717 * user_surfacing_tweet_vectors.gap_717) +
    (engaged_user_surfacing_tweet_vectors.gap_718 * user_surfacing_tweet_vectors.gap_718) +
    (engaged_user_surfacing_tweet_vectors.gap_719 * user_surfacing_tweet_vectors.gap_719) +
    (engaged_user_surfacing_tweet_vectors.gap_720 * user_surfacing_tweet_vectors.gap_720) +
    (engaged_user_surfacing_tweet_vectors.gap_721 * user_surfacing_tweet_vectors.gap_721) +
    (engaged_user_surfacing_tweet_vectors.gap_722 * user_surfacing_tweet_vectors.gap_722) +
    (engaged_user_surfacing_tweet_vectors.gap_723 * user_surfacing_tweet_vectors.gap_723) +
    (engaged_user_surfacing_tweet_vectors.gap_724 * user_surfacing_tweet_vectors.gap_724) +
    (engaged_user_surfacing_tweet_vectors.gap_725 * user_surfacing_tweet_vectors.gap_725) +
    (engaged_user_surfacing_tweet_vectors.gap_726 * user_surfacing_tweet_vectors.gap_726) +
    (engaged_user_surfacing_tweet_vectors.gap_727 * user_surfacing_tweet_vectors.gap_727) +
    (engaged_user_surfacing_tweet_vectors.gap_728 * user_surfacing_tweet_vectors.gap_728) +
    (engaged_user_surfacing_tweet_vectors.gap_729 * user_surfacing_tweet_vectors.gap_729) +
    (engaged_user_surfacing_tweet_vectors.gap_730 * user_surfacing_tweet_vectors.gap_730) +
    (engaged_user_surfacing_tweet_vectors.gap_731 * user_surfacing_tweet_vectors.gap_731) +
    (engaged_user_surfacing_tweet_vectors.gap_732 * user_surfacing_tweet_vectors.gap_732) +
    (engaged_user_surfacing_tweet_vectors.gap_733 * user_surfacing_tweet_vectors.gap_733) +
    (engaged_user_surfacing_tweet_vectors.gap_734 * user_surfacing_tweet_vectors.gap_734) +
    (engaged_user_surfacing_tweet_vectors.gap_735 * user_surfacing_tweet_vectors.gap_735) +
    (engaged_user_surfacing_tweet_vectors.gap_736 * user_surfacing_tweet_vectors.gap_736) +
    (engaged_user_surfacing_tweet_vectors.gap_737 * user_surfacing_tweet_vectors.gap_737) +
    (engaged_user_surfacing_tweet_vectors.gap_738 * user_surfacing_tweet_vectors.gap_738) +
    (engaged_user_surfacing_tweet_vectors.gap_739 * user_surfacing_tweet_vectors.gap_739) +
    (engaged_user_surfacing_tweet_vectors.gap_740 * user_surfacing_tweet_vectors.gap_740) +
    (engaged_user_surfacing_tweet_vectors.gap_741 * user_surfacing_tweet_vectors.gap_741) +
    (engaged_user_surfacing_tweet_vectors.gap_742 * user_surfacing_tweet_vectors.gap_742) +
    (engaged_user_surfacing_tweet_vectors.gap_743 * user_surfacing_tweet_vectors.gap_743) +
    (engaged_user_surfacing_tweet_vectors.gap_744 * user_surfacing_tweet_vectors.gap_744) +
    (engaged_user_surfacing_tweet_vectors.gap_745 * user_surfacing_tweet_vectors.gap_745) +
    (engaged_user_surfacing_tweet_vectors.gap_746 * user_surfacing_tweet_vectors.gap_746) +
    (engaged_user_surfacing_tweet_vectors.gap_747 * user_surfacing_tweet_vectors.gap_747) +
    (engaged_user_surfacing_tweet_vectors.gap_748 * user_surfacing_tweet_vectors.gap_748) +
    (engaged_user_surfacing_tweet_vectors.gap_749 * user_surfacing_tweet_vectors.gap_749) +
    (engaged_user_surfacing_tweet_vectors.gap_750 * user_surfacing_tweet_vectors.gap_750) +
    (engaged_user_surfacing_tweet_vectors.gap_751 * user_surfacing_tweet_vectors.gap_751) +
    (engaged_user_surfacing_tweet_vectors.gap_752 * user_surfacing_tweet_vectors.gap_752) +
    (engaged_user_surfacing_tweet_vectors.gap_753 * user_surfacing_tweet_vectors.gap_753) +
    (engaged_user_surfacing_tweet_vectors.gap_754 * user_surfacing_tweet_vectors.gap_754) +
    (engaged_user_surfacing_tweet_vectors.gap_755 * user_surfacing_tweet_vectors.gap_755) +
    (engaged_user_surfacing_tweet_vectors.gap_756 * user_surfacing_tweet_vectors.gap_756) +
    (engaged_user_surfacing_tweet_vectors.gap_757 * user_surfacing_tweet_vectors.gap_757) +
    (engaged_user_surfacing_tweet_vectors.gap_758 * user_surfacing_tweet_vectors.gap_758) +
    (engaged_user_surfacing_tweet_vectors.gap_759 * user_surfacing_tweet_vectors.gap_759) +
    (engaged_user_surfacing_tweet_vectors.gap_760 * user_surfacing_tweet_vectors.gap_760) +
    (engaged_user_surfacing_tweet_vectors.gap_761 * user_surfacing_tweet_vectors.gap_761) +
    (engaged_user_surfacing_tweet_vectors.gap_762 * user_surfacing_tweet_vectors.gap_762) +
    (engaged_user_surfacing_tweet_vectors.gap_763 * user_surfacing_tweet_vectors.gap_763) +
    (engaged_user_surfacing_tweet_vectors.gap_764 * user_surfacing_tweet_vectors.gap_764) +
    (engaged_user_surfacing_tweet_vectors.gap_765 * user_surfacing_tweet_vectors.gap_765) +
    (engaged_user_surfacing_tweet_vectors.gap_766 * user_surfacing_tweet_vectors.gap_766) +
    (engaged_user_surfacing_tweet_vectors.gap_767 * user_surfacing_tweet_vectors.gap_767)
  ) as dot_product_of_engaged_tweet_and_engaging_user_surfacing_tweets
from {table_name} t
left join user_surfacing_tweet_vectors on t.engaging_user_id = user_surfacing_tweet_vectors.user_id
left join user_surfacing_tweet_vectors as engaged_user_surfacing_tweet_vectors on t.engaged_user_id = engaged_user_surfacing_tweet_vectors.user_id
order by tweet_id, engaging_user_id
"""


if __name__ == "__main__":
    BertSimilarityBetweenEngagedAndEngagingSurfacingTweetVectorsFeature.main()
