defmodule Example.Dataset do
  @moduledoc false

  def example_data(name) do
    "#{name}.json"
    |> File.stream!()
    |> Stream.map(fn data ->
      data
      |> Jason.decode!()
      |> Enum.map(fn result ->
        book = result["book"]
        chapter = result["chapter"]
        verse = result["verse"]
        text = result["text"]
        {book, chapter, verse, text}
      end)
    end)
    |> Enum.to_list()
  end

  def gen_bible(name) do
    example_data(name)
    |> List.flatten()
    |> Enum.map(fn {book, chapter, verse, text} ->
      text = String.downcase(text)
      citation = "#{book} #{chapter}:#{verse}"
      %{citation: citation, book: book, chapter: chapter, verse: verse, text: text}
    end)
    |> List.flatten()
    |> Enum.map(fn data ->
      [data.citation, data.book, data.chapter, data.verse, data.text]
    end)
    |> Enum.shuffle()
    |> writecsv(name)
  end

  def writecsv(data, name) do
    file = File.open!("#{name}.csv", [:write, :utf8])

    data
    |> CSV.encode()
    |> Enum.each(&IO.write(file, &1))
  end
end
