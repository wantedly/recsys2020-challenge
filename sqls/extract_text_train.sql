WITH subset AS (
  SELECT tweet_id,  any_value(text_tokens) as text_tokens, any_value(tweet_type) as tweet_type, any_value(language) as language
  FROM `recsys2020.training`
  GROUP BY tweet_id
)
, decode as (
  SELECT
    tweet_id,
    any_value(language) as language,
    case
      when any_value(tweet_type) = "Retweet" then REGEXP_REPLACE(string_agg(if(bt.token like "##%", substr(bt.token, 3), bt.token), "" order by offset asc), "RT@[a-zA-Z0-9_]{1,15}:", "")
      else string_agg(if(bt.token like "##%", substr(bt.token, 3), bt.token), "" order by offset asc)
      end as text
  FROM subset
  CROSS JOIN unnest(text_tokens) as text_token with offset as offset
  INNER JOIN `recsys2020.bert_tokens` bt ON bt.id = text_token
  WHERE text_token NOT IN (100, 101, 102)   -- remove [UNK], [CLS] and [SEP]
  GROUP BY tweet_id
)

select tweet_id, language, SHA256(text) as text_id, text
from decode
