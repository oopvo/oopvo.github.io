---
layout: default
title: 名词百科
permalink: /glossary/
---

<div class="glossary-index">
  <header class="glossary-index-header">
    <h1>📚 名词百科</h1>
    <p class="glossary-index-desc">全站技术名词汇总，点击查看详细解释</p>
  </header>

  {% assign all_terms = site.glossary | sort: "title" %}

  {% if all_terms.size > 0 %}
  <div class="glossary-grid">
    {% for term in all_terms %}
    <a href="{{ site.baseurl }}{{ term.url }}" class="glossary-card-link">
      <article class="glossary-card">
        <div class="glossary-card-header">
          <h2 class="glossary-card-term">{{ term.title }}</h2>
          {% if term.categories %}
          <span class="glossary-card-cat">{{ term.categories | first }}</span>
          {% endif %}
        </div>
        <p class="glossary-card-excerpt">{{ term.content | strip_html | truncate: 120 }}</p>
        {% if term.related_posts %}
        <div class="glossary-card-source">
          📎 {{ term.related_posts | first }}
        </div>
        {% endif %}
      </article>
    </a>
    {% endfor %}
  </div>
  {% else %}
  <div class="glossary-empty">
    <p>暂无名词，敬请期待</p>
  </div>
  {% endif %}
</div>
