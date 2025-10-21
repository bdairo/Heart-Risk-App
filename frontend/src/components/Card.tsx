import type { PropsWithChildren, HTMLAttributes } from "react";

type CardProps = PropsWithChildren<HTMLAttributes<HTMLDivElement>> & {
  variant?: "default" | "emphasis";
};

export function Card({ children, className = "", variant = "default", ...rest }: CardProps) {
  const base = "rounded-xl border p-4 shadow-sm ";
  const styles = variant === "emphasis"
    ? "border-blue-200 bg-blue-50 "
    : "border-gray-200 bg-white ";
  return (
    <div className={base + styles + className} {...rest}>
      {children}
    </div>
  );
}

export function CardTitle({ children, className = "" }: PropsWithChildren<{ className?: string }>) {
  return <div className={"text-sm text-gray-500 " + className}>{children}</div>;
}

export function CardValue({ children, className = "" }: PropsWithChildren<{ className?: string }>) {
  return <div className={"mt-1 text-2xl font-semibold " + className}>{children}</div>;
}


