import type { ButtonHTMLAttributes } from "react";

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "default";
};

export function Button({ className = "", variant = "default", ...rest }: ButtonProps) {
  const base = "inline-flex items-center justify-center h-11 px-5 rounded-lg font-medium disabled:opacity-50 disabled:cursor-not-allowed ";
  const styles = variant === "primary"
    ? "bg-blue-600 text-white hover:bg-blue-700 active:bg-blue-800 "
    : "bg-white text-gray-900 border border-gray-300 hover:bg-gray-50 ";
  return <button className={base + styles + className} {...rest} />;
}


