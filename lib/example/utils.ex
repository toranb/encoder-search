defmodule Example.Utils do

  @max_length 100

  def insert_books(now) do
    books = Enum.map(1..66, fn id -> %{id: id, title: book_name(id), inserted_at: now, updated_at: now} end)
    Example.Repo.insert_all(Example.Book, books)
  end

  def seed(filename) do
    now = NaiveDateTime.utc_now() |> NaiveDateTime.truncate(:second)

    insert_books(now)

    "#{filename}.csv"
    |> File.stream!()
    |> MyParser.parse_stream()
    |> Stream.map(fn [_cit, book, chapter, verse, text] ->
      %{book: book, chapter: chapter, verse: verse, text: text}
    end)
    |> Enum.to_list()
    |> Enum.map(fn data ->
      %Example.Verse{}
      |> Example.Verse.changeset(%{
        book_id: String.to_integer(data.book),
        chapter: String.to_integer(data.chapter),
        verse: String.to_integer(data.verse),
        text: data.text
      })
    end)
    |> Enum.each(fn verse ->
      verse |> Example.Repo.insert!()
    end)
  end

  def add_embeddings() do
    verses = Example.Verse |> Example.Repo.all()
    source_texts = verses |> Enum.map(fn verse -> verse.text end)

    {:ok, tokenizer} = Tokenizers.Tokenizer.from_pretrained("bert-base-uncased")

    model_data = File.read!("#{Path.dirname(__ENV__.file)}/bible_encoder")
    deserialized_data = Nx.deserialize(model_data)
    params = deserialized_data.model

    serving =
      Example.Generation.get_embeddings(params, tokenizer,
        output_pool: :mean_pooling,
        embedding_processor: :l2_norm,
        compile: [batch_size: 512, sequence_length: [@max_length]],
        defn_options: [compiler: EXLA]
      )

    embeddings = Nx.Serving.run(serving, source_texts)

    0..(length(embeddings) - 1)
    |> Enum.map(fn i ->
      verse = Enum.at(verses, i)
      result = Enum.at(embeddings, i)
      {verse, result.embedding}
    end)
    |> Enum.each(fn {verse, embedding} ->
      verse
      |> Example.Verse.changeset(%{embedding: embedding})
      |> Example.Repo.update!()
    end)
  end

  @bible_books %{
    1 => "Genesis",
    2 => "Exodus",
    3 => "Leviticus",
    4 => "Numbers",
    5 => "Deuteronomy",
    6 => "Joshua",
    7 => "Judges",
    8 => "Ruth",
    9 => "1 Samuel",
    10 => "2 Samuel",
    11 => "1 Kings",
    12 => "2 Kings",
    13 => "1 Chronicles",
    14 => "2 Chronicles",
    15 => "Ezra",
    16 => "Nehemiah",
    17 => "Esther",
    18 => "Job",
    19 => "Psalms",
    20 => "Proverbs",
    21 => "Ecclesiastes",
    22 => "Song of Solomon",
    23 => "Isaiah",
    24 => "Jeremiah",
    25 => "Lamentations",
    26 => "Ezekiel",
    27 => "Daniel",
    28 => "Hosea",
    29 => "Joel",
    30 => "Amos",
    31 => "Obadiah",
    32 => "Jonah",
    33 => "Micah",
    34 => "Nahum",
    35 => "Habakkuk",
    36 => "Zephaniah",
    37 => "Haggai",
    38 => "Zechariah",
    39 => "Malachi",
    40 => "Matthew",
    41 => "Mark",
    42 => "Luke",
    43 => "John",
    44 => "Acts",
    45 => "Romans",
    46 => "1 Corinthians",
    47 => "2 Corinthians",
    48 => "Galatians",
    49 => "Ephesians",
    50 => "Philippians",
    51 => "Colossians",
    52 => "1 Thessalonians",
    53 => "2 Thessalonians",
    54 => "1 Timothy",
    55 => "2 Timothy",
    56 => "Titus",
    57 => "Philemon",
    58 => "Hebrews",
    59 => "James",
    60 => "1 Peter",
    61 => "2 Peter",
    62 => "1 John",
    63 => "2 John",
    64 => "3 John",
    65 => "Jude",
    66 => "Revelation"
  }

  def book_name(number) do
    Map.get(@bible_books, number, "Unknown Book")
  end
end
