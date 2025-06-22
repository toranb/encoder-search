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
      limit: 4
    )
    |> Example.Repo.all()
  end
end
