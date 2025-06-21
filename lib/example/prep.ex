defmodule Example.Prep do
  NimbleCSV.define(MyParser, separator: ",", escape: "\"")

  @context_size 90

  def generate(filename, opts \\ []) do
    shuffle_examples = Keyword.get(opts, :shuffle, false)
    encoder_training? = Keyword.get(opts, :encoder, false)

    "#{filename}.csv"
    |> File.stream!()
    |> MyParser.parse_stream()
    |> Stream.map(fn [cit, book, _chapter, _verse, text] ->
      lower_cit = String.downcase(cit) |> String.trim()
      lower_book = String.downcase(book) |> String.trim()
      lower_text = String.downcase(text) |> String.trim()

      text =
        if String.ends_with?(lower_text, ".") && !encoder_training? do
          lower_text <> " SEP"
        else
          lower_text
        end

      %{book: lower_book, cit: lower_cit, text: text}
    end)
    |> Enum.to_list()
    |> Enum.uniq_by(& &1.text)
    |> combine_short_verses(@context_size)
    |> Enum.flat_map(&generate_training_data(&1.text, @context_size))
    |> then(&if shuffle_examples, do: &1 |> Enum.shuffle(), else: &1)
    |> Enum.map(fn data ->
      context = data.context |> Enum.join(" ")
      [context, data.target]
    end)
    |> Enum.shuffle()
    |> writecsv("#{filename}training")
  end

  def generate_training_data(verse_text, window_size) do
    tokens = tokenize(verse_text)

    if length(tokens) >= window_size + 1 do
      0..(length(tokens) - (window_size + 1))
      |> Enum.map(fn i ->
        context = Enum.slice(tokens, i, window_size)
        target = Enum.at(tokens, i + window_size)
        %{context: context, target: target}
      end)
    else
      []
    end
  end

  def writecsv(data, name) do
    file = File.open!("#{name}.csv", [:write, :utf8])

    data
    |> CSV.encode()
    |> Enum.each(&IO.write(file, &1))
  end

  def combine_short_verses(verses, target_words \\ 100) do
    verses
    |> Enum.reduce({[], []}, fn verse, {windows, current_window} ->
      process_verse(verse, current_window, windows, target_words)
    end)
    |> finalize_windows()
  end

  defp process_verse(verse, current_window, windows, target_words) do
    verse_words = count_words(verse.text)
    current_word_count = get_current_word_count(current_window)

    cond do
      current_window == [] ->
        {windows, [verse]}

      current_word_count + verse_words > target_words * 1.4 and current_word_count >= target_words * 0.8 ->
        window = create_window(current_window)
        {[window | windows], [verse]}

      current_word_count >= target_words and current_word_count + verse_words > target_words * 1.2 ->
        window = create_window(current_window)
        {[window | windows], [verse]}

      true ->
        {windows, current_window ++ [verse]}
    end
  end

  defp finalize_windows({windows, current_window}) do
    final_windows = case current_window do
      [] -> windows
      verses -> [create_window(verses) | windows]
    end

    Enum.reverse(final_windows)
  end

  defp create_window(verses) do
    combined_text =
      verses
      |> Enum.map(& &1.text)
      |> Enum.join(" ")

    citations = Enum.map(verses, & &1.cit)

    %{
      text: combined_text,
      word_count: count_words(combined_text),
      verses: citations,
      verse_count: length(verses)
    }
  end

  defp get_current_word_count(verses) when verses == [], do: 0
  defp get_current_word_count(verses) do
    verses
    |> Enum.map(&count_words(&1.text))
    |> Enum.sum()
  end

  defp count_words(text) do
    text
    |> String.trim()
    |> String.split(~r/\s+/)
    |> length()
  end

  def display_windows(windows) do
    windows
    |> Enum.with_index(1)
    |> Enum.each(fn {window, index} ->
      IO.puts("=== Window #{index} ===")
      IO.puts("Word count: #{window.word_count}")
      IO.puts("Verses: #{Enum.join(window.verses, ", ")}")
      IO.puts("Text: #{window.text}")
      IO.puts("")
    end)
  end

  defp tokenize(text) do
    text
    |> String.split(~r/\s+/)
    |> Enum.map(&String.trim/1)
    |> Enum.filter(&(byte_size(&1) > 0))
  end

  def shuffle() do
    "pretraining.csv"
    |> File.stream!()
    |> EncoderTokenParser.parse_stream(skip_headers: false)
    |> Stream.map(fn [text, target] ->
      %{text: text, target: target}
    end)
    |> Enum.to_list()
    |> Enum.uniq_by(& &1.text)
    |> Enum.map(fn %{text: text, target: target} ->
      [text, target]
    end)
    |> Enum.shuffle()
    |> writecsv("training")
  end
end
