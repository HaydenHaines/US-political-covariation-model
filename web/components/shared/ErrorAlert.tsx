"use client";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { AlertCircle } from "lucide-react";

interface ErrorAlertProps {
  title: string;
  message: string;
  retry?: () => void;
}

export function ErrorAlert({ title, message, retry }: ErrorAlertProps) {
  return (
    <Alert variant="destructive">
      <AlertCircle className="h-4 w-4" />
      <AlertTitle>{title}</AlertTitle>
      <AlertDescription>
        {message}
        {retry && (
          <button
            onClick={retry}
            className="ml-2 underline hover:no-underline focus:outline-none"
          >
            Try again
          </button>
        )}
      </AlertDescription>
    </Alert>
  );
}
