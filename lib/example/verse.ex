defmodule Example.Verse do
  use Ecto.Schema

  import Ecto.Query
  import Ecto.Changeset
  import Pgvector.Ecto.Query

  alias __MODULE__

  schema "verses" do
    field(:chapter, :integer)
    field(:verse, :integer)
    field(:text, :string)
    field(:embedding, Pgvector.Ecto.Vector)

    belongs_to(:book, Example.Book)

    timestamps()
  end

  @required_attrs [:chapter, :verse, :text, :book_id]
  @optional_attrs [:embedding]

  def changeset(verse, params \\ %{}) do
    verse
    |> cast(params, @required_attrs ++ @optional_attrs)
    |> validate_required(@required_attrs)
  end

  def search(book_id, embedding) do
    from(v in Verse,
      select: {v.id, v.chapter, v.verse, v.text, v.book_id},
      where: v.book_id == ^book_id,
      order_by: cosine_distance(v.embedding, ^embedding),
      limit: 25
    )
    |> Example.Repo.all()
  end

  def search_keywords(_document_id, term) do
    sql = """
    SELECT
      bm.score,
      bm.verse_id,
      v.chapter,
      v.verse,
      bm.content as text,
      v.book_id
    FROM search_verses($1) bm
    INNER JOIN verse_stats d ON d.verse_id = bm.verse_id
    INNER JOIN verses v ON v.id = bm.verse_id
    ORDER BY bm.score DESC;
    """

    case Ecto.Adapters.SQL.query(Example.Repo, sql, [term]) do
      {:ok, %{rows: rows}} ->
        Enum.map(rows, fn [score, verse_id, chapter, verse, text, book_id] ->
          {score, {verse_id, chapter, verse, text, book_id}}
        end)

      {:error, _error} ->
        []
    end
  end

  def index_verses() do
    sql = """
    SELECT FROM index_all_verses()
    """
    Ecto.Adapters.SQL.query(Example.Repo, sql, [])
  end

  def reindex_verses() do
    sql = """
    SELECT FROM bulk_update_modified_verses()
    """
    Ecto.Adapters.SQL.query(Example.Repo, sql, [])
  end
end
