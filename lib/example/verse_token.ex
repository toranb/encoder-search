defmodule Example.VerseToken do
  use Ecto.Schema

  import Ecto.Query
  import Ecto.Changeset
  import Pgvector.Ecto.Query

  alias __MODULE__

  schema "verse_tokens" do
    field(:token_pos, :integer)
    field(:embedding, Pgvector.Ecto.Vector)

    belongs_to(:verse, Example.Verse)

    timestamps()
  end

  @required_attrs [:token_pos, :verse_id, :embedding]

  def changeset(verse_token, params \\ %{}) do
    verse_token
    |> cast(params, @required_attrs)
    |> validate_required(@required_attrs)
  end

  def search(verse_ids, embeddings, opts \\ []) do
    id_list =
      case verse_ids do
        [] -> nil
        ids -> ids |> Enum.map(&to_string/1) |> Enum.join(",")
      end

    id_filter = "v.id IN (#{id_list})"
    opts = Keyword.put(opts, :candidate_filter, id_filter)
    colbert_search(embeddings, opts)
  end

  def colbert_search(embeddings, opts \\ []) do
    limit = Keyword.get(opts, :limit, 10)
    candidate_filter = Keyword.get(opts, :candidate_filter, nil)

    query_embeddings =
      embeddings
      |> Nx.to_list()
      |> Enum.with_index(1)
      |> Enum.map(fn {embedding, token_pos} ->
        embedding_str = "[#{Enum.join(embedding, ",")}]"
        {token_pos, embedding_str}
      end)

    query_token_cte =
      query_embeddings
      |> Enum.map(fn {token_pos, embedding_str} ->
        "SELECT #{token_pos} as token_pos, '#{embedding_str}'::vector as embedding"
      end)
      |> Enum.join(" UNION ALL ")

    filter_clause =
      case candidate_filter do
        nil -> ""
        filter -> "WHERE #{filter}"
      end

    sql = """
    WITH query_tokens AS (
      #{query_token_cte}
    ),
    maxsim_scores AS (
      SELECT
        v.id as verse_id,
        v.chapter,
        v.verse,
        v.text,
        v.book_id,
        qt.token_pos,
        MAX(1 - (qt.embedding <=> vt.embedding)) as max_sim
      FROM query_tokens qt
      CROSS JOIN verses v
      INNER JOIN verse_tokens vt ON v.id = vt.verse_id
      #{filter_clause}
      GROUP BY v.id, v.chapter, v.verse, v.text, v.book_id, qt.token_pos
    )
    SELECT
      verse_id,
      chapter,
      verse,
      text,
      book_id,
      SUM(max_sim) as colbert_score
    FROM maxsim_scores
    GROUP BY verse_id, chapter, verse, text, book_id
    ORDER BY colbert_score DESC
    LIMIT $1;
    """

    case Ecto.Adapters.SQL.query(Example.Repo, sql, [limit]) do
      {:ok, %{rows: rows}} ->
        Enum.map(rows, fn [verse_id, chapter, verse, text, book_id, score] ->
          {score, {verse_id, chapter, verse, text, book_id}}
        end)

      {:error, _error} ->
        []
    end
  end
end
