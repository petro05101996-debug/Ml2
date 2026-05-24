import React from "react";
export default function ReportSection({ title, children }: { title: string; children?: React.ReactNode }) { return <section><h2>{title}</h2>{children}</section>; }
