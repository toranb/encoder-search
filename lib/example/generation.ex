defmodule Example.Generation do
  @moduledoc false

  alias Example.Encoder
  alias Bumblebee.Shared

  @max_length 100
  @pad_token_id 1

  def get_embeddings(params, tokenizer, opts \\ []) do
    opts =
      Keyword.validate!(opts, [
        :compile,
        output_pool: nil,
        embedding_processor: nil,
        defn_options: [],
        preallocate_params: false
      ])

    output_pool = opts[:output_pool]
    embedding_processor = opts[:embedding_processor]
    preallocate_params = opts[:preallocate_params]
    defn_options = opts[:defn_options]

    compile =
      if compile = opts[:compile] do
        compile
        |> Keyword.validate!([:batch_size, :sequence_length])
        |> Shared.require_options!([:batch_size, :sequence_length])
      end

    batch_size = compile[:batch_size]
    sequence_length = compile[:sequence_length]
    base_key = Nx.Random.key(1234)
    keys = Nx.Random.split(base_key, parts: 20)

    embedding_fun = fn params, inputs ->
      output = Encoder.encoder_forward(params, inputs["input_ids"], keys, training: false)

      output =
        case output_pool do
          nil ->
            output

          :mean_pooling ->
            input_mask_expanded = Nx.new_axis(inputs["attention_mask"], -1)

            output
            |> Nx.multiply(input_mask_expanded)
            |> Nx.sum(axes: [1])
            |> Nx.divide(Nx.sum(input_mask_expanded, axes: [1]))

          other ->
            raise ArgumentError,
                  "expected :output_pool to be :mean_pooling or nil, got: #{inspect(other)}"
        end

      output =
        case embedding_processor do
          nil ->
            output

          :l2_norm ->
            Bumblebee.Utils.Nx.normalize(output)

          other ->
            raise ArgumentError,
                  "expected :embedding_processor to be one of nil or :l2_norm, got: #{inspect(other)}"
        end

      output
    end

    batch_keys = Shared.sequence_batch_keys(sequence_length)

    Nx.Serving.new(
      fn batch_key, defn_options ->
        params = Shared.maybe_preallocate(params, preallocate_params, defn_options)

        scope = {:embedding, batch_key}

        embedding_fun =
          Shared.compile_or_jit(embedding_fun, scope, defn_options, compile != nil, fn ->
            {:sequence_length, sequence_length} = batch_key

            inputs = %{
              "input_ids" => Nx.template({batch_size, sequence_length}, :u32),
              "attention_mask" => Nx.template({batch_size, sequence_length}, :u32)
            }

            [params, inputs]
          end)

        fn inputs ->
          inputs = Shared.maybe_pad(inputs, batch_size)
          embedding_fun.(params, inputs) |> Shared.serving_post_computation()
        end
      end,
      defn_options
    )
    |> Nx.Serving.batch_size(batch_size)
    |> Nx.Serving.process_options(batch_keys: batch_keys)
    |> Nx.Serving.client_preprocessing(fn input ->
      {texts, multi?} = Shared.validate_serving_input!(input, &Shared.validate_string/1)

      inputs =
        Nx.with_default_backend(Nx.BinaryBackend, fn ->
          token_ids =
            texts
            |> Enum.map(fn text ->
              {:ok, encoding} = Tokenizers.Tokenizer.encode(tokenizer, text)
              bert_text_ids = Tokenizers.Encoding.get_ids(encoding)
              pre_text_ids = Enum.slice(bert_text_ids, 1..-2//1)
              Enum.take(pre_text_ids, @max_length)
            end)

          max_length = token_ids |> Enum.map(&length/1) |> Enum.max(fn -> 0 end)
          padded_token_ids = Example.Encoder.pad_tokens(token_ids, max_length, @pad_token_id)

          input_tensor =
            Nx.tensor(padded_token_ids, backend: Nx.BinaryBackend) |> Nx.as_type(:u32)

          input_mask = Nx.not_equal(input_tensor, @pad_token_id) |> Nx.as_type(:u32)

          %{
            "input_ids" => input_tensor,
            "attention_mask" => input_mask
          }
        end)

      batch_key = Shared.sequence_batch_key_for_inputs(inputs, sequence_length)
      batch = [inputs] |> Nx.Batch.concatenate() |> Nx.Batch.key(batch_key)

      {batch, multi?}
    end)
    |> Nx.Serving.client_postprocessing(fn {embeddings, _metadata}, multi? ->
      for embedding <- Bumblebee.Utils.Nx.batch_to_list(embeddings) do
        %{embedding: embedding}
      end
      |> Shared.normalize_output(multi?)
    end)
  end
end
