import Link from "next/link";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";

interface EntryPoint {
  title: string;
  description: string;
  href: string;
}

const ENTRY_POINTS: EntryPoint[] = [
  {
    title: "See the full forecast \u2192",
    description:
      "Senate race-by-race predictions with interactive maps and poll integration.",
    href: "/forecast",
  },
  {
    title: "Explore electoral types \u2192",
    description:
      "Discover the community types that drive American elections.",
    href: "/types",
  },
  {
    title: "How the model works \u2192",
    description:
      "Type discovery, covariance structure, and Bayesian poll propagation explained.",
    href: "/methodology",
  },
];

export function EntryPoints() {
  return (
    <section className="mx-auto grid max-w-3xl grid-cols-1 gap-4 px-4 py-8 sm:grid-cols-3">
      {ENTRY_POINTS.map((entry) => (
        <Link
          key={entry.href}
          href={entry.href}
          className="no-underline"
        >
          <Card className="h-full transition-colors hover:bg-[var(--color-surface-raised)]">
            <CardHeader>
              <CardTitle className="text-base">{entry.title}</CardTitle>
              <CardDescription>{entry.description}</CardDescription>
            </CardHeader>
          </Card>
        </Link>
      ))}
    </section>
  );
}
