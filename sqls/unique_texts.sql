select tweet_id, any_value(text_tokens) as text_tokens
from (
    select tweet_id, any_value(text_tokens) as text_tokens
    from `recsys2020.training`
    group by tweet_id
    union all
    select tweet_id, any_value(text_tokens) as text_tokens
    from `recsys2020.test`
    group by tweet_id
)
group by tweet_id
