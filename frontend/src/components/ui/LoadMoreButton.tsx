import { Button } from "./Button";

export function LoadMoreButton({
  onClick,
  loading,
}: {
  onClick: () => void;
  loading: boolean;
}) {
  return (
    <div className="flex justify-center mt-4">
      <Button variant="secondary" onClick={onClick} disabled={loading}>
        {loading ? "Loading…" : "Load More"}
      </Button>
    </div>
  );
}
