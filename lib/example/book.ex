defmodule Example.Book do
  use Ecto.Schema

  import Ecto.Changeset

  schema "books" do
    field(:title, :string)

    has_many(:verses, Example.Verse, preload_order: [asc: :inserted_at])

    timestamps()
  end

  @required_attrs [:title]

  def changeset(book, params \\ %{}) do
    book
    |> cast(params, @required_attrs)
    |> validate_required(@required_attrs)
  end
end
