defmodule Example.Encoder do
  import Nx.Defn

  @warmup_steps 12_420
  @batch_size 64
  @val_batch_size 128
  @max_token_ids 104
  @attn_dims 768
  @num_heads 12
  @dropout_rate 0.143
  @pad_token_id 1
  @vocab_size 30_522
  @epsilon 1.0e-8

  NimbleCSV.define(EncoderTokenParser, separator: ",", escape: "\"")

  def get_examples() do
    "training.csv"
    |> File.stream!()
    |> EncoderTokenParser.parse_stream(skip_headers: false)
    |> Stream.map(fn [text, target] ->
      {text, target}
    end)
    |> Enum.to_list()
  end

  def scheduled(num_epochs, base_learning_rate) do
    {:ok, tokenizer} = Tokenizers.Tokenizer.from_pretrained("bert-base-uncased")

    examples = get_examples()
    {_, new_model, _opt} = train(examples, tokenizer, num_epochs, base_learning_rate)

    new_model_params = %{model: new_model}
    new_serialized_container = Nx.serialize(new_model_params)
    File.write!("#{Path.dirname(__ENV__.file)}/warm_bible_encoder", new_serialized_container)
  end

  def preprocess_examples(examples, tokenizer) do
    masking_probability = 0.22

    examples
    |> Enum.with_index()
    |> Enum.map(fn {{rawtext, _}, _index} ->
      text = String.downcase(rawtext)
      {:ok, encoding} = Tokenizers.Tokenizer.encode(tokenizer, text)
      bert_text_ids = Tokenizers.Encoding.get_ids(encoding)
      pre_text_ids = Enum.slice(bert_text_ids, 1..-2//1)
      text_ids = Enum.take(pre_text_ids, @max_token_ids)

      num_tokens = length(text_ids)
      num_tokens_to_mask = max(2, floor(num_tokens * masking_probability))
      shuffled_indices = Enum.shuffle(0..(num_tokens - 1))
      indices_to_mask_set = MapSet.new(Enum.take(shuffled_indices, num_tokens_to_mask))

      input_ids =
        text_ids
        |> Enum.with_index()
        |> Enum.map(fn {token_id, index} ->
          if MapSet.member?(indices_to_mask_set, index) do
            rand_val = :rand.uniform_real()

            cond do
              rand_val < 0.8 ->
                103

              rand_val < 0.9 ->
                :rand.uniform(@vocab_size) - 1

              true ->
                token_id
            end
          else
            token_id
          end
        end)

      target_token_ids =
        input_ids
        |> Enum.with_index()
        |> Enum.map(fn {_token_id, index} ->
          if MapSet.member?(indices_to_mask_set, index), do: 1.0, else: 0.0
        end)

      {input_ids, text_ids, target_token_ids}
    end)
  end

  defn initialize_embeddings(opts \\ []) do
    embedding_dim = opts[:embedding_dim]
    vocab_size = opts[:vocab_size]
    key = Nx.Random.key(94)
    Nx.Random.normal_split(key, 0.0, 0.02, shape: {vocab_size, embedding_dim})
  end

  defn initialize_dense(opts \\ []) do
    input_dim = opts[:input_dim]
    output_dim = opts[:output_dim]
    hidden_dim = opts[:hidden_dim]

    one_key_a = Nx.Random.key(39)
    one_key_b = Nx.Random.key(40)
    two_key_a = Nx.Random.key(41)
    two_key_b = Nx.Random.key(42)
    three_key_a = Nx.Random.key(43)
    three_key_b = Nx.Random.key(44)
    four_key_a = Nx.Random.key(45)
    four_key_b = Nx.Random.key(46)
    five_key_a = Nx.Random.key(47)
    five_key_b = Nx.Random.key(48)
    six_key_a = Nx.Random.key(49)
    six_key_b = Nx.Random.key(50)
    seven_key_a = Nx.Random.key(571)
    seven_key_b = Nx.Random.key(572)
    eight_key_a = Nx.Random.key(581)
    eight_key_b = Nx.Random.key(582)
    nine_key_a = Nx.Random.key(686)
    nine_key_b = Nx.Random.key(685)
    ten_key_a = Nx.Random.key(777)
    ten_key_b = Nx.Random.key(778)
    eleven_key_a = Nx.Random.key(877)
    eleven_key_b = Nx.Random.key(878)

    one_w1 = Nx.Random.normal_split(one_key_a, 0.0, 0.02, shape: {input_dim, hidden_dim})
    one_b1 = Nx.broadcast(0.0, {hidden_dim})
    one_w2 = Nx.Random.normal_split(one_key_b, 0.0, 0.02, shape: {hidden_dim, output_dim})
    one_b2 = Nx.broadcast(0.0, {output_dim})

    two_w1 = Nx.Random.normal_split(two_key_a, 0.0, 0.02, shape: {input_dim, hidden_dim})
    two_b1 = Nx.broadcast(0.0, {hidden_dim})
    two_w2 = Nx.Random.normal_split(two_key_b, 0.0, 0.02, shape: {hidden_dim, output_dim})
    two_b2 = Nx.broadcast(0.0, {output_dim})

    three_w1 = Nx.Random.normal_split(three_key_a, 0.0, 0.02, shape: {input_dim, hidden_dim})
    three_b1 = Nx.broadcast(0.0, {hidden_dim})
    three_w2 = Nx.Random.normal_split(three_key_b, 0.0, 0.02, shape: {hidden_dim, output_dim})
    three_b2 = Nx.broadcast(0.0, {output_dim})

    four_w1 = Nx.Random.normal_split(four_key_a, 0.0, 0.02, shape: {input_dim, hidden_dim})
    four_b1 = Nx.broadcast(0.0, {hidden_dim})
    four_w2 = Nx.Random.normal_split(four_key_b, 0.0, 0.02, shape: {hidden_dim, output_dim})
    four_b2 = Nx.broadcast(0.0, {output_dim})

    five_w1 = Nx.Random.normal_split(five_key_a, 0.0, 0.02, shape: {input_dim, hidden_dim})
    five_b1 = Nx.broadcast(0.0, {hidden_dim})
    five_w2 = Nx.Random.normal_split(five_key_b, 0.0, 0.02, shape: {hidden_dim, output_dim})
    five_b2 = Nx.broadcast(0.0, {output_dim})

    six_w1 = Nx.Random.normal_split(six_key_a, 0.0, 0.02, shape: {input_dim, hidden_dim})
    six_b1 = Nx.broadcast(0.0, {hidden_dim})
    six_w2 = Nx.Random.normal_split(six_key_b, 0.0, 0.02, shape: {hidden_dim, output_dim})
    six_b2 = Nx.broadcast(0.0, {output_dim})

    seven_w1 = Nx.Random.normal_split(seven_key_a, 0.0, 0.02, shape: {input_dim, hidden_dim})
    seven_b1 = Nx.broadcast(0.0, {hidden_dim})
    seven_w2 = Nx.Random.normal_split(seven_key_b, 0.0, 0.02, shape: {hidden_dim, output_dim})
    seven_b2 = Nx.broadcast(0.0, {output_dim})

    eight_w1 = Nx.Random.normal_split(eight_key_a, 0.0, 0.02, shape: {input_dim, hidden_dim})
    eight_b1 = Nx.broadcast(0.0, {hidden_dim})
    eight_w2 = Nx.Random.normal_split(eight_key_b, 0.0, 0.02, shape: {hidden_dim, output_dim})
    eight_b2 = Nx.broadcast(0.0, {output_dim})

    nine_w1 = Nx.Random.normal_split(nine_key_a, 0.0, 0.02, shape: {input_dim, hidden_dim})
    nine_b1 = Nx.broadcast(0.0, {hidden_dim})
    nine_w2 = Nx.Random.normal_split(nine_key_b, 0.0, 0.02, shape: {hidden_dim, output_dim})
    nine_b2 = Nx.broadcast(0.0, {output_dim})

    ten_w1 = Nx.Random.normal_split(ten_key_a, 0.0, 0.02, shape: {input_dim, hidden_dim})
    ten_b1 = Nx.broadcast(0.0, {hidden_dim})
    ten_w2 = Nx.Random.normal_split(ten_key_b, 0.0, 0.02, shape: {hidden_dim, output_dim})
    ten_b2 = Nx.broadcast(0.0, {output_dim})

    eleven_w1 = Nx.Random.normal_split(eleven_key_a, 0.0, 0.02, shape: {input_dim, hidden_dim})
    eleven_b1 = Nx.broadcast(0.0, {hidden_dim})
    eleven_w2 = Nx.Random.normal_split(eleven_key_b, 0.0, 0.02, shape: {hidden_dim, output_dim})
    eleven_b2 = Nx.broadcast(0.0, {output_dim})

    {one_w1, one_b1, one_w2, one_b2, two_w1, two_b1, two_w2, two_b2, three_w1, three_b1, three_w2,
     three_b2, four_w1, four_b1, four_w2, four_b2, five_w1, five_b1, five_w2, five_b2, six_w1,
     six_b1, six_w2, six_b2, seven_w1, seven_b1, seven_w2, seven_b2, eight_w1, eight_b1, eight_w2, eight_b2, nine_w1, nine_b1, nine_w2, nine_b2, ten_w1, ten_b1, ten_w2, ten_b2, eleven_w1, eleven_b1, eleven_w2, eleven_b2}
  end

  defn initialize_mlm_output_weights(opts \\ []) do
    input_dim = opts[:input_dim]
    vocab_size = opts[:vocab_size]
    key = Nx.Random.key(51)

    out_w = Nx.Random.normal_split(key, 0.0, 0.02, shape: {input_dim, vocab_size})
    out_b = Nx.broadcast(0.0, {vocab_size})

    {out_w, out_b}
  end

  defn initialize_attention(opts \\ []) do
    dims = opts[:dims]
    attn_dims = opts[:attn_dims]

    one_key_a = Nx.Random.key(11)
    one_key_b = Nx.Random.key(12)
    one_key_c = Nx.Random.key(13)
    one_key_d = Nx.Random.key(14)

    one_w_query = Nx.Random.normal_split(one_key_a, 0.0, 0.02, shape: {dims, attn_dims})
    one_w_key = Nx.Random.normal_split(one_key_b, 0.0, 0.02, shape: {dims, attn_dims})
    one_w_value = Nx.Random.normal_split(one_key_c, 0.0, 0.02, shape: {dims, attn_dims})
    one_w_out = Nx.Random.normal_split(one_key_d, 0.0, 0.02, shape: {attn_dims, dims})

    two_key_a = Nx.Random.key(15)
    two_key_b = Nx.Random.key(16)
    two_key_c = Nx.Random.key(17)
    two_key_d = Nx.Random.key(18)

    two_w_query = Nx.Random.normal_split(two_key_a, 0.0, 0.02, shape: {dims, attn_dims})
    two_w_key = Nx.Random.normal_split(two_key_b, 0.0, 0.02, shape: {dims, attn_dims})
    two_w_value = Nx.Random.normal_split(two_key_c, 0.0, 0.02, shape: {dims, attn_dims})
    two_w_out = Nx.Random.normal_split(two_key_d, 0.0, 0.02, shape: {attn_dims, dims})

    three_key_a = Nx.Random.key(19)
    three_key_b = Nx.Random.key(20)
    three_key_c = Nx.Random.key(21)
    three_key_d = Nx.Random.key(22)

    three_w_query = Nx.Random.normal_split(three_key_a, 0.0, 0.02, shape: {dims, attn_dims})
    three_w_key = Nx.Random.normal_split(three_key_b, 0.0, 0.02, shape: {dims, attn_dims})
    three_w_value = Nx.Random.normal_split(three_key_c, 0.0, 0.02, shape: {dims, attn_dims})
    three_w_out = Nx.Random.normal_split(three_key_d, 0.0, 0.02, shape: {attn_dims, dims})

    four_key_a = Nx.Random.key(23)
    four_key_b = Nx.Random.key(24)
    four_key_c = Nx.Random.key(25)
    four_key_d = Nx.Random.key(26)

    four_w_query = Nx.Random.normal_split(four_key_a, 0.0, 0.02, shape: {dims, attn_dims})
    four_w_key = Nx.Random.normal_split(four_key_b, 0.0, 0.02, shape: {dims, attn_dims})
    four_w_value = Nx.Random.normal_split(four_key_c, 0.0, 0.02, shape: {dims, attn_dims})
    four_w_out = Nx.Random.normal_split(four_key_d, 0.0, 0.02, shape: {attn_dims, dims})

    five_key_a = Nx.Random.key(27)
    five_key_b = Nx.Random.key(28)
    five_key_c = Nx.Random.key(29)
    five_key_d = Nx.Random.key(30)

    five_w_query = Nx.Random.normal_split(five_key_a, 0.0, 0.02, shape: {dims, attn_dims})
    five_w_key = Nx.Random.normal_split(five_key_b, 0.0, 0.02, shape: {dims, attn_dims})
    five_w_value = Nx.Random.normal_split(five_key_c, 0.0, 0.02, shape: {dims, attn_dims})
    five_w_out = Nx.Random.normal_split(five_key_d, 0.0, 0.02, shape: {attn_dims, dims})

    six_key_a = Nx.Random.key(31)
    six_key_b = Nx.Random.key(32)
    six_key_c = Nx.Random.key(33)
    six_key_d = Nx.Random.key(34)

    six_w_query = Nx.Random.normal_split(six_key_a, 0.0, 0.02, shape: {dims, attn_dims})
    six_w_key = Nx.Random.normal_split(six_key_b, 0.0, 0.02, shape: {dims, attn_dims})
    six_w_value = Nx.Random.normal_split(six_key_c, 0.0, 0.02, shape: {dims, attn_dims})
    six_w_out = Nx.Random.normal_split(six_key_d, 0.0, 0.02, shape: {attn_dims, dims})

    seven_key_a = Nx.Random.key(371)
    seven_key_b = Nx.Random.key(372)
    seven_key_c = Nx.Random.key(373)
    seven_key_d = Nx.Random.key(374)

    seven_w_query = Nx.Random.normal_split(seven_key_a, 0.0, 0.02, shape: {dims, attn_dims})
    seven_w_key = Nx.Random.normal_split(seven_key_b, 0.0, 0.02, shape: {dims, attn_dims})
    seven_w_value = Nx.Random.normal_split(seven_key_c, 0.0, 0.02, shape: {dims, attn_dims})
    seven_w_out = Nx.Random.normal_split(seven_key_d, 0.0, 0.02, shape: {attn_dims, dims})

    eight_key_a = Nx.Random.key(381)
    eight_key_b = Nx.Random.key(382)
    eight_key_c = Nx.Random.key(383)
    eight_key_d = Nx.Random.key(384)

    eight_w_query = Nx.Random.normal_split(eight_key_a, 0.0, 0.02, shape: {dims, attn_dims})
    eight_w_key = Nx.Random.normal_split(eight_key_b, 0.0, 0.02, shape: {dims, attn_dims})
    eight_w_value = Nx.Random.normal_split(eight_key_c, 0.0, 0.02, shape: {dims, attn_dims})
    eight_w_out = Nx.Random.normal_split(eight_key_d, 0.0, 0.02, shape: {attn_dims, dims})

    nine_key_a = Nx.Random.key(981)
    nine_key_b = Nx.Random.key(982)
    nine_key_c = Nx.Random.key(983)
    nine_key_d = Nx.Random.key(984)

    nine_w_query = Nx.Random.normal_split(nine_key_a, 0.0, 0.02, shape: {dims, attn_dims})
    nine_w_key = Nx.Random.normal_split(nine_key_b, 0.0, 0.02, shape: {dims, attn_dims})
    nine_w_value = Nx.Random.normal_split(nine_key_c, 0.0, 0.02, shape: {dims, attn_dims})
    nine_w_out = Nx.Random.normal_split(nine_key_d, 0.0, 0.02, shape: {attn_dims, dims})

    ten_key_a = Nx.Random.key(1081)
    ten_key_b = Nx.Random.key(1082)
    ten_key_c = Nx.Random.key(1083)
    ten_key_d = Nx.Random.key(1084)

    ten_w_query = Nx.Random.normal_split(ten_key_a, 0.0, 0.02, shape: {dims, attn_dims})
    ten_w_key = Nx.Random.normal_split(ten_key_b, 0.0, 0.02, shape: {dims, attn_dims})
    ten_w_value = Nx.Random.normal_split(ten_key_c, 0.0, 0.02, shape: {dims, attn_dims})
    ten_w_out = Nx.Random.normal_split(ten_key_d, 0.0, 0.02, shape: {attn_dims, dims})

    eleven_key_a = Nx.Random.key(1181)
    eleven_key_b = Nx.Random.key(1182)
    eleven_key_c = Nx.Random.key(1183)
    eleven_key_d = Nx.Random.key(1184)

    eleven_w_query = Nx.Random.normal_split(eleven_key_a, 0.0, 0.02, shape: {dims, attn_dims})
    eleven_w_key = Nx.Random.normal_split(eleven_key_b, 0.0, 0.02, shape: {dims, attn_dims})
    eleven_w_value = Nx.Random.normal_split(eleven_key_c, 0.0, 0.02, shape: {dims, attn_dims})
    eleven_w_out = Nx.Random.normal_split(eleven_key_d, 0.0, 0.02, shape: {attn_dims, dims})

    {one_w_query, one_w_key, one_w_value, one_w_out, two_w_query, two_w_key, two_w_value,
     two_w_out, three_w_query, three_w_key, three_w_value, three_w_out, four_w_query, four_w_key,
     four_w_value, four_w_out, five_w_query, five_w_key, five_w_value, five_w_out, six_w_query,
     six_w_key, six_w_value, six_w_out, seven_w_query, seven_w_key, seven_w_value, seven_w_out,
     eight_w_query, eight_w_key, eight_w_value, eight_w_out,
     nine_w_query, nine_w_key, nine_w_value, nine_w_out,
     ten_w_query, ten_w_key, ten_w_value, ten_w_out,
     eleven_w_query, eleven_w_key, eleven_w_value, eleven_w_out}
  end

  deftransformp sqrt_2_over_pi() do
    :math.sqrt(2.0 / :math.pi())
  end

  defn gelu(x) do
    x_cubed = Nx.pow(x, 3.0)
    inner_expr = Nx.add(x, Nx.multiply(0.044715, x_cubed))
    sqrt_2_pi = sqrt_2_over_pi()
    tanh_input = Nx.multiply(sqrt_2_pi, inner_expr)
    tanh_result = Nx.tanh(tanh_input)
    one_plus_tanh = Nx.add(1.0, tanh_result)
    result = Nx.multiply(0.5, Nx.multiply(x, one_plus_tanh))

    custom_grad(
      result,
      [x],
      fn g ->
        sech_squared = Nx.subtract(1.0, Nx.pow(tanh_result, 2.0))
        tanh_derivative = Nx.multiply(sqrt_2_pi, Nx.add(1.0, Nx.multiply(3.0 * 0.044715, Nx.pow(x, 2.0))))

        gelu_derivative = Nx.add(
          Nx.multiply(0.5, one_plus_tanh),
          Nx.multiply(0.5, Nx.multiply(x, Nx.multiply(sech_squared, tanh_derivative)))
        )

        [Nx.multiply(g, gelu_derivative)]
      end
    )
  end

  defn layer_norm(x) do
    mean = Nx.mean(x, axes: [-1], keep_axes: true)
    variance = Nx.variance(x, axes: [-1], keep_axes: true, ddof: 0)
    one = Nx.subtract(x, mean)
    one_b = Nx.add(variance, @epsilon)
    two = Nx.sqrt(one_b)
    Nx.divide(one, two)
  end

  defn softmax(logits) do
    max_logit = Nx.reduce_max(logits, axes: [-1], keep_axes: true)
    shifted_logits = Nx.subtract(logits, max_logit)
    exp_logits = Nx.exp(shifted_logits)
    sum_exp_logits = Nx.sum(exp_logits, axes: [-1], keep_axes: true)
    Nx.divide(exp_logits, sum_exp_logits)
  end

  defn log_softmax(logits) do
    max_logit = Nx.reduce_max(logits, axes: [-1], keep_axes: true)
    shifted_logits = Nx.subtract(logits, max_logit)
    sum_exp_shifted = Nx.sum(Nx.exp(shifted_logits), axes: [-1], keep_axes: true)
    log_sum_exp = Nx.log(sum_exp_shifted)
    Nx.subtract(shifted_logits, log_sum_exp)
  end

  defn sinusoidal_position_encoding(seq_len, dim) do
    positions = Nx.iota({seq_len, 1})
    i_indices = Nx.iota({dim})

    angle_one = Nx.multiply(i_indices, 2.0)
    angle_two = Nx.divide(angle_one, dim)
    angle_three = Nx.pow(10000.0, angle_two)
    angle_rates = Nx.divide(1.0, angle_three)
    angle_reshaped = Nx.reshape(angle_rates, {1, dim})

    angles = Nx.dot(positions, angle_reshaped)
    even_mask = Nx.equal(Nx.remainder(i_indices, 2), 0)
    sin_vals = Nx.sin(angles)
    cos_vals = Nx.cos(angles)
    target_shape = Nx.shape(sin_vals)
    broadcasted_mask = Nx.broadcast(even_mask, target_shape)
    Nx.select(broadcasted_mask, sin_vals, cos_vals)
  end

  defn self_attention(input, w_query, w_key, w_value, w_out, attention_mask) do
    num_heads = @num_heads
    input_shape = Nx.shape(input)
    batch_size = elem(input_shape, 0)
    seq_len = elem(input_shape, 1)
    projection_dim = @attn_dims
    head_dim = div(projection_dim, num_heads)

    query = Nx.dot(input, w_query)
    key = Nx.dot(input, w_key)
    value = Nx.dot(input, w_value)

    q = query |> Nx.reshape({batch_size, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    k = key |> Nx.reshape({batch_size, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    v = value |> Nx.reshape({batch_size, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    attention_scores = Nx.dot(q, [3], [0, 1], k, [3], [0, 1])
    scaling_divisor = Nx.sqrt(head_dim)
    scaled_attention_scores = Nx.divide(attention_scores, scaling_divisor)
    additive_mask = Nx.select(attention_mask, 0.0, -1.0e8)
    masked_scores = Nx.add(scaled_attention_scores, additive_mask)
    attention_weights = softmax(masked_scores)
    context = Nx.dot(attention_weights, [3], [0, 1], v, [2], [0, 1])
    context = Nx.reshape(context, {batch_size, seq_len, projection_dim})
    Nx.dot(context, w_out)
  end

  defn create_attention_mask(input_ids) do
    {batch_size, seq_len} = Nx.shape(input_ids)
    mask = Nx.not_equal(input_ids, @pad_token_id)
    mask_2d = Nx.new_axis(mask, 1)
    mask_2d_broadcast = Nx.broadcast(mask_2d, {batch_size, seq_len, seq_len})
    Nx.new_axis(mask_2d_broadcast, 1)
  end

  defn feed_forward_network(input, w1, b1, w2, b2) do
    input |> Nx.dot(w1) |> Nx.add(b1) |> gelu() |> Nx.dot(w2) |> Nx.add(b2)
  end

  defn residual_connection(input, output) do
    Nx.add(input, output)
  end

  defn dropout(input, key, training) do
    if training do
      mask_shape = Nx.shape(input)
      random_vals = Nx.Random.normal_split(key, 0.0, 1.0, shape: mask_shape)
      keep_mask = Nx.greater(random_vals, @dropout_rate)
      keep_prob = Nx.subtract(1.0, @dropout_rate)
      scale_factor = Nx.divide(1.0, keep_prob)
      input |> Nx.multiply(keep_mask) |> Nx.multiply(scale_factor)
    else
      input
    end
  end

  defn encoder_forward(
         {embeddings,
          {one_w_query, one_w_key, one_w_value, one_w_out, two_w_query, two_w_key, two_w_value,
           two_w_out, three_w_query, three_w_key, three_w_value, three_w_out, four_w_query,
           four_w_key, four_w_value, four_w_out, five_w_query, five_w_key, five_w_value,
           five_w_out, six_w_query, six_w_key, six_w_value, six_w_out, seven_w_query, seven_w_key,
            seven_w_value, seven_w_out, eight_w_query, eight_w_key, eight_w_value, eight_w_out, nine_w_query, nine_w_key, nine_w_value, nine_w_out,
            ten_w_query, ten_w_key, ten_w_value, ten_w_out,
            eleven_w_query, eleven_w_key, eleven_w_value, eleven_w_out},
          {one_w1, one_b1, one_w2, one_b2, two_w1, two_b1, two_w2, two_b2, three_w1, three_b1,
           three_w2, three_b2, four_w1, four_b1, four_w2, four_b2, five_w1, five_b1, five_w2,
           five_b2, six_w1, six_b1, six_w2, six_b2, seven_w1, seven_b1, seven_w2, seven_b2,
            eight_w1, eight_b1, eight_w2, eight_b2, nine_w1, nine_b1, nine_w2, nine_b2,
            ten_w1, ten_b1, ten_w2, ten_b2, eleven_w1, eleven_b1, eleven_w2, eleven_b2}, _out_w, _out_b},
          input_ids,
          keys,
          opts \\ []
       ) do
    training = opts[:training]

    key_one = keys[0]
    key_one_two = keys[1]
    key_two = keys[2]
    key_two_two = keys[3]
    key_three = keys[4]
    key_three_two = keys[5]
    key_four = keys[6]
    key_four_two = keys[7]
    key_five = keys[8]
    key_five_two = keys[9]
    key_six = keys[10]
    key_six_two = keys[11]
    key_seven = keys[12]
    key_seven_two = keys[13]
    key_eight = keys[14]
    key_eight_two = keys[15]
    key_nine = keys[16]
    key_nine_two = keys[17]
    key_ten = keys[18]
    key_ten_two = keys[19]
    key_eleven = keys[20]
    key_eleven_two = keys[21]

    attention_mask = create_attention_mask(input_ids)

    word_embeddings = embeddings |> Nx.take(input_ids)
    {_batch_size, seq_len, dim} = Nx.shape(word_embeddings)
    positional_embeddings = sinusoidal_position_encoding(seq_len, dim)
    x = Nx.add(word_embeddings, positional_embeddings)

    # Layer 1
    norm_attn_one = layer_norm(x)
    attn_out_one = self_attention(norm_attn_one, one_w_query, one_w_key, one_w_value, one_w_out, attention_mask) |> dropout(key_one, training)
    add_one = residual_connection(x, attn_out_one)
    norm_ffn_one = layer_norm(add_one)
    ffn_out_one = feed_forward_network(norm_ffn_one, one_w1, one_b1, one_w2, one_b2) |> dropout(key_one_two, training)
    one = residual_connection(add_one, ffn_out_one)

    # Layer 2
    norm_attn_two = layer_norm(one)
    attn_out_two = self_attention(norm_attn_two, two_w_query, two_w_key, two_w_value, two_w_out, attention_mask) |> dropout(key_two, training)
    add_two = residual_connection(one, attn_out_two)
    norm_ffn_two = layer_norm(add_two)
    ffn_out_two = feed_forward_network(norm_ffn_two, two_w1, two_b1, two_w2, two_b2) |> dropout(key_two_two, training)
    two = residual_connection(add_two, ffn_out_two)

    # Layer 3
    norm_attn_three = layer_norm(two)
    attn_out_three = self_attention(norm_attn_three, three_w_query, three_w_key, three_w_value, three_w_out, attention_mask) |> dropout(key_three, training)
    add_three = residual_connection(two, attn_out_three)
    norm_ffn_three = layer_norm(add_three)
    ffn_out_three = feed_forward_network(norm_ffn_three, three_w1, three_b1, three_w2, three_b2) |> dropout(key_three_two, training)
    three = residual_connection(add_three, ffn_out_three)

    # Layer 4
    norm_attn_four = layer_norm(three)
    attn_out_four = self_attention(norm_attn_four, four_w_query, four_w_key, four_w_value, four_w_out, attention_mask) |> dropout(key_four, training)
    add_four = residual_connection(three, attn_out_four)
    norm_ffn_four = layer_norm(add_four)
    ffn_out_four = feed_forward_network(norm_ffn_four, four_w1, four_b1, four_w2, four_b2) |> dropout(key_four_two, training)
    four = residual_connection(add_four, ffn_out_four)

    # Layer 5
    norm_attn_five = layer_norm(four)
    attn_out_five = self_attention(norm_attn_five, five_w_query, five_w_key, five_w_value, five_w_out, attention_mask) |> dropout(key_five, training)
    add_five = residual_connection(four, attn_out_five)
    norm_ffn_five = layer_norm(add_five)
    ffn_out_five = feed_forward_network(norm_ffn_five, five_w1, five_b1, five_w2, five_b2) |> dropout(key_five_two, training)
    five = residual_connection(add_five, ffn_out_five)

    # Layer 6
    norm_attn_six = layer_norm(five)
    attn_out_six = self_attention(norm_attn_six, six_w_query, six_w_key, six_w_value, six_w_out, attention_mask) |> dropout(key_six, training)
    add_six = residual_connection(five, attn_out_six)
    norm_ffn_six = layer_norm(add_six)
    ffn_out_six = feed_forward_network(norm_ffn_six, six_w1, six_b1, six_w2, six_b2) |> dropout(key_six_two, training)
    six = residual_connection(add_six, ffn_out_six)

    # Layer 7
    norm_attn_seven = layer_norm(six)
    attn_out_seven = self_attention(norm_attn_seven, seven_w_query, seven_w_key, seven_w_value, seven_w_out, attention_mask) |> dropout(key_seven, training)
    add_seven = residual_connection(six, attn_out_seven)
    norm_ffn_seven = layer_norm(add_seven)
    ffn_out_seven = feed_forward_network(norm_ffn_seven, seven_w1, seven_b1, seven_w2, seven_b2) |> dropout(key_seven_two, training)
    seven = residual_connection(add_seven, ffn_out_seven)

    # Layer 8
    norm_attn_eight = layer_norm(seven)
    attn_out_eight = self_attention(norm_attn_eight, eight_w_query, eight_w_key, eight_w_value, eight_w_out, attention_mask) |> dropout(key_eight, training)
    add_eight = residual_connection(seven, attn_out_eight)
    norm_ffn_eight = layer_norm(add_eight)
    ffn_out_eight = feed_forward_network(norm_ffn_eight, eight_w1, eight_b1, eight_w2, eight_b2) |> dropout(key_eight_two, training)
    eight = residual_connection(add_eight, ffn_out_eight)

    # Layer 9
    norm_attn_nine = layer_norm(eight)
    attn_out_nine = self_attention(norm_attn_nine, nine_w_query, nine_w_key, nine_w_value, nine_w_out, attention_mask) |> dropout(key_nine, training)
    add_nine = residual_connection(eight, attn_out_nine)
    norm_ffn_nine = layer_norm(add_nine)
    ffn_out_nine = feed_forward_network(norm_ffn_nine, nine_w1, nine_b1, nine_w2, nine_b2) |> dropout(key_nine_two, training)
    nine = residual_connection(add_nine, ffn_out_nine)

    # Layer 10
    norm_attn_ten = layer_norm(nine)
    attn_out_ten = self_attention(norm_attn_ten, ten_w_query, ten_w_key, ten_w_value, ten_w_out, attention_mask) |> dropout(key_ten, training)
    add_ten = residual_connection(nine, attn_out_ten)
    norm_ffn_ten = layer_norm(add_ten)
    ffn_out_ten = feed_forward_network(norm_ffn_ten, ten_w1, ten_b1, ten_w2, ten_b2) |> dropout(key_ten_two, training)
    ten = residual_connection(add_ten, ffn_out_ten)

    # Layer 11
    norm_attn_eleven = layer_norm(ten)
    attn_out_eleven = self_attention(norm_attn_eleven, eleven_w_query, eleven_w_key, eleven_w_value, eleven_w_out, attention_mask) |> dropout(key_eleven, training)
    add_eleven = residual_connection(ten, attn_out_eleven)
    norm_ffn_eleven = layer_norm(add_eleven)
    ffn_out_eleven = feed_forward_network(norm_ffn_eleven, eleven_w1, eleven_b1, eleven_w2, eleven_b2) |> dropout(key_eleven_two, training)
    eleven = residual_connection(add_eleven, ffn_out_eleven)

    # Final LayerNorm
    layer_norm(eleven)
  end

  defn loss({_, _, _, out_w, out_b} = params, input_ids, target_tensor, mask_pos_tensor, base_key, opts \\ []) do
    training = opts[:training]
    keys = Nx.Random.split(base_key, parts: 22)

    out_logits = encoder_forward(params, input_ids, keys, training: training)
    all_logits = out_logits |> Nx.dot(out_w) |> Nx.add(out_b)
    {batch_size, seq_len, _} = Nx.shape(all_logits)

    log_probs = log_softmax(all_logits)
    target_indices_for_gather = Nx.reshape(target_tensor, {batch_size, seq_len, 1})
    log_probs_of_true_tokens = Nx.take_along_axis(log_probs, target_indices_for_gather, axis: 2)
    squeezed_log_probs_of_true_tokens = Nx.squeeze(log_probs_of_true_tokens, axes: [2])

    per_token_neg_log_likelihood = Nx.multiply(squeezed_log_probs_of_true_tokens, -1.0)
    masked_token_losses = Nx.multiply(per_token_neg_log_likelihood, mask_pos_tensor)

    num_masked_tokens = Nx.sum(mask_pos_tensor)
    sum_loss = Nx.sum(masked_token_losses)
    safe_divisor = Nx.max(num_masked_tokens, Nx.tensor(1.0))
    Nx.divide(sum_loss, safe_divisor)
  end

  defn update(params, optimizer_state, input_ids, target_id, mask_pos, update_fn, base_key) do
    {loss, gradient} = value_and_grad(params, &loss(&1, input_ids, target_id, mask_pos, base_key, training: true))
    {scaled_updates, new_optimizer_state} = update_fn.(gradient, optimizer_state, params)
    {Polaris.Updates.apply_updates(params, scaled_updates), new_optimizer_state, loss}
  end

  def train(examples, tokenizer, num_epochs, base_learning_rate) do
    dims = 768
    embed_dims = 768
    hidden_dims = 3_072
    preprocessed_examples = preprocess_examples(examples, tokenizer)
    blocksize = get_blocksize(preprocessed_examples)

    embeddings = initialize_embeddings(vocab_size: @vocab_size, embedding_dim: embed_dims)

    attn_weights = initialize_attention(dims: dims, attn_dims: @attn_dims, num_heads: @num_heads)
    dense_weights = initialize_dense(input_dim: dims, hidden_dim: hidden_dims, output_dim: dims)
    {out_w, out_b} = initialize_mlm_output_weights(input_dim: dims, vocab_size: @vocab_size)

    initial_params = {embeddings, attn_weights, dense_weights, out_w, out_b}

    num_training_examples = floor(length(examples) * 0.9)
    steps_per_epoch = ceil(num_training_examples / @batch_size)
    total_steps = steps_per_epoch * num_epochs
    neg_base_learning_rate = -base_learning_rate
    warmup_steps = @warmup_steps

    schedule_fn = fn count ->
      result_one = Nx.add(count, 1)
      result_two = Nx.divide(result_one, warmup_steps)
      warmup_rate = Nx.multiply(neg_base_learning_rate, result_two)
      progress_top = Nx.subtract(count, warmup_steps)
      progress_bottom = Nx.subtract(total_steps, warmup_steps)
      progress = Nx.divide(progress_top, progress_bottom)
      decay_rate_result = Nx.subtract(1, progress)
      decay_rate = Nx.multiply(neg_base_learning_rate, decay_rate_result)
      Nx.select(Nx.less(count, warmup_steps), warmup_rate, decay_rate)
    end

    {init_fn, update_fn} =
      Polaris.Updates.clip_by_global_norm(max_norm: 1.0)
      |> Polaris.Updates.scale_by_adam()
      |> Polaris.Updates.add_decayed_weights(decay: 0.01)
      |> Polaris.Updates.scale_by_schedule(schedule_fn)

    init_optimizer_state = init_fn.(initial_params)

    example_len = floor(length(preprocessed_examples) * 0.9)
    {training_data, validation_data} = Enum.split(preprocessed_examples, example_len)

    Enum.reduce(1..num_epochs, {0, initial_params, init_optimizer_state}, fn epoch, {global_idx, epoch_params, optimizer_state} ->
      shuffled_examples = Enum.shuffle(training_data)
      batches = Enum.chunk_every(shuffled_examples, @batch_size)
      batch_count = Enum.count(batches)
      steps_per_batch = @batch_size * batch_count

      {_, final_epoch_params, final_opt_state, total_loss} =
        batches
        |> Enum.reduce({0, epoch_params, optimizer_state, 0.0}, fn thebatch, {next_idx, params, opt_state, acc_loss} ->
          batch = Enum.shuffle(thebatch)
          batch_id_list = batch |> Enum.map(fn {ids, _, _} -> ids end)
          input_ids = pad_tokens(batch_id_list, blocksize, @pad_token_id)
          input_tensor = Nx.tensor(input_ids)

          tar_id = batch |> Enum.map(fn {_, targ_id, _} -> targ_id end)
          pad_tar_ids = pad_tokens(tar_id, blocksize, @pad_token_id)
          target_tensor = Nx.tensor(pad_tar_ids)

          mask_pos = batch |> Enum.map(fn {_, _, masked} -> masked end)
          padded_float_mask = pad_tokens(mask_pos, blocksize, 0.0)
          mask_pos_tensor = Nx.tensor(padded_float_mask)

          base_key = Nx.Random.key(next_idx)
          {new_params, new_optimizer_state, example_loss} = update(params, opt_state, input_tensor, target_tensor, mask_pos_tensor, update_fn, base_key)

          loss_value = Nx.to_number(example_loss)

          if next_idx > 0 do
            Nx.backend_deallocate(params)
            Nx.backend_deallocate(opt_state)
          end

          {next_idx + @batch_size, new_params, new_optimizer_state, acc_loss + loss_value}
        end)

      base_key = Nx.Random.key(global_idx)

      if rem(epoch, 2) == 0 do
        gradient_info = diagnose_learning_issues(final_epoch_params, shuffled_examples, base_key)
        gradient_info.mean_gradient_norm |> IO.inspect(label: "Epoch #{epoch} with Mean Gradient Norm", limit: :infinity)
      end

      if rem(epoch, 2) == 0 do
        shuffled_validation_examples = Enum.shuffle(validation_data)
        validate_batch = Enum.take(shuffled_validation_examples, @val_batch_size)

        batch_id_list = validate_batch |> Enum.map(fn {ids, _, _} -> ids end)
        input_ids = pad_tokens(batch_id_list, blocksize, @pad_token_id)
        val_input_tensor = Nx.tensor(input_ids)

        tar_id = validate_batch |> Enum.map(fn {_, targ_id, _} -> targ_id end)
        pad_tar_ids = pad_tokens(tar_id, blocksize, @pad_token_id)
        val_target_tensor = Nx.tensor(pad_tar_ids)

        mask_pos = validate_batch |> Enum.map(fn {_, _, masked} -> masked end)
        padded_float_mask = pad_tokens(mask_pos, blocksize, 0.0)
        val_mask_pos_tensor = Nx.tensor(padded_float_mask)

        validation_loss = loss(final_epoch_params, val_input_tensor, val_target_tensor, val_mask_pos_tensor, base_key, training: false)
        val_loss_number = Nx.to_number(validation_loss)

        avg_epoch_loss = total_loss / length(batches)

        eval_results = get_evals() |> Enum.map(fn text -> predict(final_epoch_params, text, tokenizer) end)
        eval_average = Enum.sum(eval_results) / length(eval_results)
        eval_average |> IO.inspect(label: "Epoch #{epoch} with EVAL AVG", limit: :infinity)

        ## checkpoint
        checkpoint_params = %{model: final_epoch_params}
        checkpoint_container = Nx.serialize(checkpoint_params)
        File.write!("#{Path.dirname(__ENV__.file)}/bible_encoder_checkpoint_#{epoch}", checkpoint_container)

        Nx.backend_deallocate(val_input_tensor)
        Nx.backend_deallocate(val_target_tensor)
        Nx.backend_deallocate(val_mask_pos_tensor)

        IO.puts("Epoch #{epoch}, Train Loss: #{avg_epoch_loss}, Validation Loss: #{val_loss_number}")
      end

      if epoch_params != initial_params do
        Nx.backend_deallocate(epoch_params)
        Nx.backend_deallocate(optimizer_state)
      end

      {global_idx + steps_per_batch, final_epoch_params, final_opt_state}
    end)
  end

  def pad_tokens(lists, max_length, value) do
    Enum.map(lists, fn list ->
      current_length = length(list)
      padding_needed = max_length - current_length

      if padding_needed > 0 do
        padding = List.duplicate(value, padding_needed)
        list ++ padding
      else
        list
      end
    end)
  end

  def get_blocksize(examples) do
    examples
    |> Enum.map(fn {input_ids, _, _} ->
      input_ids
    end)
    |> Enum.map(fn token_ids ->
      length(token_ids)
    end)
    |> Enum.max(fn -> 0 end)
  end

  def predict({_, _, _, out_w, out_b} = params, text, tokenizer) do
    batch = inf_preprocess_examples([text], tokenizer)
    blocksize = get_blocksize(batch)

    batch_id_list = batch |> Enum.map(fn {ids, _, _} -> ids end)
    input_ids = pad_tokens(batch_id_list, blocksize, @pad_token_id)
    input_tensor = Nx.tensor(input_ids)
    tar_id = batch |> Enum.map(fn {_, targ_id, _} -> targ_id end)
    pad_tar_ids = pad_tokens(tar_id, blocksize, @pad_token_id)
    mask_pos = batch |> Enum.map(fn {_, _, masked} -> masked end)
    padded_float_mask = pad_tokens(mask_pos, blocksize, 0.0)

    base_key = Nx.Random.key(1234)
    keys = Nx.Random.split(base_key, parts: 22)
    out_logits = encoder_forward(params, input_tensor, keys, training: false)
    all_logits = out_logits |> Nx.dot(out_w) |> Nx.add(out_b)

    predicted_ids_tensor = Nx.argmax(all_logits, axis: -1)
    predicted_ids = Nx.to_flat_list(predicted_ids_tensor)
    [padded_targets] = pad_tar_ids
    [padded_positions] = padded_float_mask

    total_results =
      Enum.zip([predicted_ids, padded_targets, padded_positions])
      |> Enum.map(fn {token, predicted, pos} ->
        %{pos: pos, token: token, result: predicted}
      end)
      |> Enum.filter(&(&1.pos == 1.0))

    total = Enum.count(total_results)

    matches =
      Enum.reduce(total_results, 0, fn map, acc ->
        if map.result == map.token do
          acc + 1
        else
          acc
        end
      end)

    percentage = matches / total * 100
    Float.round(percentage, 2)
  end

  def inf_preprocess_examples(examples, tokenizer) do
    masking_probability = 0.12

    examples
    |> Enum.with_index()
    |> Enum.map(fn {text, _index} ->
      {:ok, encoding} = Tokenizers.Tokenizer.encode(tokenizer, text)
      bert_text_ids = Tokenizers.Encoding.get_ids(encoding)
      pre_text_ids = Enum.slice(bert_text_ids, 1..-2//1)
      text_ids = Enum.take(pre_text_ids, @max_token_ids)

      num_tokens = length(text_ids)
      example_len = floor(num_tokens * masking_probability)
      num_tokens_to_mask = max(1, example_len)
      shuffled_indices = Enum.shuffle(0..(num_tokens - 1))
      indices_to_mask_set = MapSet.new(Enum.take(shuffled_indices, num_tokens_to_mask))

      input_ids =
        text_ids
        |> Enum.with_index()
        |> Enum.map(fn {token_id, index} ->
          if MapSet.member?(indices_to_mask_set, index) do
            103
          else
            token_id
          end
        end)

      masked_positions =
        input_ids
        |> Enum.map(fn token_id -> if token_id == 103, do: 1.0, else: 0.0 end)

      {input_ids, text_ids, masked_positions}
    end)
  end

  def get_evals() do
    "evals.csv"
    |> File.stream!()
    |> EncoderTokenParser.parse_stream(skip_headers: false)
    |> Stream.map(fn [text, _target] ->
      text
    end)
    |> Enum.to_list()
  end

  defn get_gradients(params, input_ids, target_id, mask_pos, base_key) do
    {_, gradient} = value_and_grad(params, &loss(&1, input_ids, target_id, mask_pos, base_key, training: true))
    gradient
  end

  def diagnose_learning_issues(model_params, examples, base_key) do
    preprocessed = Enum.take(examples, @batch_size)
    blocksize = get_blocksize(preprocessed)

    gradient_norms = Enum.map(preprocessed, fn {batch_id_list, tar_id, mask_pos} ->
      input_ids = pad_tokens([batch_id_list], blocksize, @pad_token_id)
      input_tensor = Nx.tensor(input_ids)

      pad_tar_ids = pad_tokens([tar_id], blocksize, @pad_token_id)
      target_tensor = Nx.tensor(pad_tar_ids)

      padded_float_mask = pad_tokens([mask_pos], blocksize, 0.0)
      mask_pos_tensor = Nx.tensor(padded_float_mask)

      gradient = get_gradients(model_params, input_tensor, target_tensor, mask_pos_tensor, base_key)

      total_norm = calculate_gradient_norm(gradient) |> Nx.to_number()

      Nx.backend_deallocate(gradient)

      total_norm
    end)

    %{
      mean_gradient_norm: Enum.sum(gradient_norms) / length(gradient_norms),
      gradient_norms: gradient_norms
    }
  end

  def calculate_gradient_norm(gradients) do
    sum_gs = deep_reduce_gradients(gradients, Nx.tensor(0.0))
    Nx.sqrt(sum_gs)
  end

  def deep_reduce_gradients(gradients, acc) do
    case gradients do
      %Nx.Tensor{} = tensor ->
        tensor
        |> Nx.pow(2)
        |> Nx.sum()
        |> Nx.add(acc)

      gradients when is_tuple(gradients) ->
        gradients
        |> Tuple.to_list()
        |> Enum.reduce(acc, fn grad_element, current_acc ->
          deep_reduce_gradients(grad_element, current_acc)
        end)

      gradients when is_map(gradients) ->
        gradients
        |> Map.values()
        |> Enum.reduce(acc, fn grad_element, current_acc ->
          deep_reduce_gradients(grad_element, current_acc)
        end)

      gradients when is_list(gradients) ->
        gradients
        |> Enum.reduce(acc, fn grad_element, current_acc ->
          deep_reduce_gradients(grad_element, current_acc)
        end)
    end
  end

  def generate(params, text, tokenizer) do
    {:ok, encoding} = Tokenizers.Tokenizer.encode(tokenizer, text)
    bert_text_ids = Tokenizers.Encoding.get_ids(encoding)
    pre_text_ids = Enum.slice(bert_text_ids, 1..-2//1)
    text_ids = Enum.take(pre_text_ids, @max_token_ids)
    blocksize = [text_ids] |> Enum.map(fn tokens -> length(tokens) end) |> Enum.max(fn -> 0 end)
    input_ids = pad_tokens([text_ids], blocksize, @pad_token_id)
    input_tensor = Nx.tensor(input_ids)
    base_key = Nx.Random.key(1234)
    keys = Nx.Random.split(base_key, parts: 22)
    encoder_forward(params, input_tensor, keys, training: false)
  end
end
